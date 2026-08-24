"""Orchestration for explanation generation.

Everything that touches a database lives here. The generation package it calls holds no session and
imports no ORM, so the split is structural: content generation *cannot* write to the ledger.

Nothing in this module opens a transaction that touches ``risk_decisions`` or ``review_items``.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.dashboard.schemas import EntityContext, ExplanationView
from riskloom.db.models import RiskDecision as RiskDecisionRow
from riskloom.db.models import RiskDecisionExplanation as ExplanationRow
from riskloom.explanations import factors
from riskloom.explanations.client import GeminiClientProtocol, GeminiError
from riskloom.explanations.prompt import PROMPT_VERSION, render_facts
from riskloom.explanations.sanitizer import verify
from riskloom.explanations.schemas import (
    EntityAggregate,
    ExplanationInput,
    ExplanationRejected,
    FactorCode,
    FailSafeReasonInput,
)
from riskloom.services.dashboard import SelfInclusion, entity_context_for

MAX_ATTEMPTS_PER_DECISION = 3

logger = structlog.get_logger("riskloom.explanations")


@dataclass
class ExplanationBudget:
    """Process-level cap on Gemini calls.

    Copies :class:`riskloom.services.preflight.OrderBudget` including its semantics: the unit is
    taken before the outbound request, so a call that then fails still consumes one. What needs
    bounding is outbound calls to a third party, and counting only successes would let an endlessly
    failing path retry without limit.
    """

    limit: int
    attempted: int = 0

    def take(self) -> bool:
        if self.attempted >= self.limit:
            return False
        self.attempted += 1
        return True


class ExplanationRefused(Exception):
    """A request that must not proceed. ``code`` maps to an HTTP status at the route."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def is_eligible(row: RiskDecisionRow) -> bool:
    """Only a finalised DENY carries a risk narrative worth generating.

    REVIEW is never a risk band in this system: its four causes are operational fail-safes, and a
    REVIEW row's ``risk_decision`` is either NULL or ``allow``. Generating prose about
    ``order_budget_exhausted`` would describe an internal quota counter in the register of a risk
    finding, which is worse than saying nothing.

    ``status == 'final'`` is defence in depth rather than a fix. Preflight sets ``risk_decision``
    and ``status = 'final'`` inside one transaction, so a ``pending`` row cannot carry a deny
    verdict today. This predicate is exactly where that assumption would silently break under a
    future refactor, so it is asserted rather than assumed.
    """

    return row.status == "final" and row.risk_decision == "deny"


def _probability_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def build_input(row: RiskDecisionRow, context: list[EntityContext]) -> ExplanationInput:
    """Project a ledger row into the allowlisted fact set.

    Raw tokens are dropped here and cannot be recovered downstream: ``EntityAggregate`` has no
    field to hold one.
    """

    probability = _probability_text(row.calibrated_probability)
    threshold = _probability_text(row.decision_threshold)
    if probability is None or threshold is None or row.risk_decision is None:
        raise ExplanationRefused("not_eligible")

    return ExplanationInput(
        calibrated_probability=probability,
        decision_threshold=threshold,
        probability_exceeds_threshold=Decimal(probability) >= Decimal(threshold),
        risk_decision=row.risk_decision,  # type: ignore[arg-type]
        action=row.action or "review",  # type: ignore[arg-type]
        fail_safe_reason=(
            FailSafeReasonInput(row.fail_safe_reason) if row.fail_safe_reason else None
        ),
        amount_subunits=row.amount_subunits,
        currency=row.currency,
        channel=row.channel,  # type: ignore[arg-type]
        context=[
            EntityAggregate(
                kind=item.kind,
                present=item.token is not None,
                decision_count=item.decision_count,
                denied_count=item.denied_count,
                review_count=item.review_count,
                span_seconds=item.span_seconds,
            )
            for item in context
        ],
    )


def input_digest(payload: ExplanationInput) -> str:
    """SHA-256 over the exact canonical facts sent upstream."""

    return hashlib.sha256(render_facts(payload).encode("utf-8")).hexdigest()


def to_view(row: ExplanationRow, attempts_used: int) -> ExplanationView:
    codes = [FactorCode(code) for code in (row.factors or [])]
    return ExplanationView(
        status=row.status,  # type: ignore[arg-type]
        summary=row.summary,
        factors=[factors.render(code) for code in codes],
        factor_codes=[code.value for code in codes],
        caveat=row.caveat,
        failure_reason=row.failure_reason,
        model_name=row.model_name,
        prompt_version=row.prompt_version,
        attempt_number=row.attempt_number,
        attempts_remaining=max(MAX_ATTEMPTS_PER_DECISION - attempts_used, 0),
        created_at=row.created_at,
    )


async def _attempts(session: AsyncSession, decision_id: UUID) -> list[ExplanationRow]:
    result = await session.scalars(
        select(ExplanationRow)
        .where(ExplanationRow.risk_decision_id == decision_id)
        .order_by(ExplanationRow.attempt_number)
    )
    return list(result)


async def latest(session: AsyncSession, decision_id: UUID) -> ExplanationView | None:
    """The current explanation state for a decision, if any attempt exists."""

    rows = await _attempts(session, decision_id)
    if not rows:
        return None
    ready = [row for row in rows if row.status == "ready"]
    chosen = ready[-1] if ready else rows[-1]
    return to_view(chosen, attempts_used=len(rows))


async def _decision(session: AsyncSession, decision_id: UUID) -> RiskDecisionRow:
    row = await session.get(RiskDecisionRow, decision_id)
    if row is None:
        raise ExplanationRefused("not_found")
    return row


async def generate(
    session: AsyncSession,
    client: GeminiClientProtocol | None,
    budget: ExplanationBudget,
    decision_id: UUID,
    model_name: str,
) -> ExplanationView:
    """Claim an attempt, call the model, store the outcome.

    The claim happens before any outbound call, so a concurrent duplicate request is refused rather
    than spending a second unit of budget on the same decision.
    """

    row = await _decision(session, decision_id)
    if not is_eligible(row):
        raise ExplanationRefused("not_eligible")

    existing = await _attempts(session, decision_id)
    if any(item.status == "pending" for item in existing):
        raise ExplanationRefused("in_progress")
    if any(item.status == "ready" for item in existing):
        return await _require_view(session, decision_id)
    if len(existing) >= MAX_ATTEMPTS_PER_DECISION:
        raise ExplanationRefused("attempts_exhausted")
    if client is None:
        raise ExplanationRefused("not_configured")

    # EXCLUDE_SELF is mandatory here, not a preference. Every factor this evidence can support is
    # phrased as prior history, and the decision being explained is already committed by the time
    # this runs. Counting it let a rotated card-testing instrument -- one whose only appearance in
    # the ledger is this very attempt -- support "prior denials on this instrument".
    context = await entity_context_for(session, row, SelfInclusion.EXCLUDE_SELF)
    payload = build_input(row, context)
    digest = input_digest(payload)
    attempt_number = len(existing) + 1

    # Those reads autobegan an implicit transaction, and an explicit ``begin()`` on top of one
    # raises. Everything needed downstream has already been copied into ``payload``, which is a
    # plain Pydantic object holding no session state, so the read transaction is closed here.
    await session.rollback()

    claim = ExplanationRow(
        risk_decision_id=decision_id,
        attempt_number=attempt_number,
        status="pending",
        model_name=model_name,
        prompt_version=PROMPT_VERSION,
        input_digest=digest,
    )
    try:
        async with session.begin():
            session.add(claim)
    except IntegrityError as exc:
        raise ExplanationRefused("in_progress") from exc

    # The budget is taken after the claim and before the call, so a refused duplicate never spends
    # a unit and an attempt that fails upstream still does.
    if not budget.take():
        await _finalise(session, claim, status="failed", reason="budget_exhausted")
        raise ExplanationRefused("budget_exhausted")

    status, reason, explanation = "ready", None, None
    try:
        explanation = verify(await client.explain(payload), payload)
    except GeminiError as exc:
        status, reason = "failed", str(exc)
    except ExplanationRejected as exc:
        status, reason = "rejected", str(exc)

    await _finalise(session, claim, status=status, reason=reason, explanation=explanation)
    logger.info(
        "explanation_generated",
        decision_id=str(decision_id),
        attempt_number=attempt_number,
        status=status,
        failure_reason=reason,
        model_name=model_name,
    )
    return await _require_view(session, decision_id)


async def _finalise(
    session: AsyncSession,
    claim: ExplanationRow,
    *,
    status: str,
    reason: str | None,
    explanation: object | None = None,
) -> None:
    """The single pending -> terminal transition. Touches no other table."""

    async with session.begin():
        claim.status = status
        claim.completed_at = datetime.now(UTC)
        claim.failure_reason = reason
        if explanation is not None:
            claim.summary = explanation.summary  # type: ignore[attr-defined]
            claim.factors = [code.value for code in explanation.factors]  # type: ignore[attr-defined]
            claim.caveat = explanation.caveat  # type: ignore[attr-defined]
        session.add(claim)


async def _require_view(session: AsyncSession, decision_id: UUID) -> ExplanationView:
    view = await latest(session, decision_id)
    if view is None:  # pragma: no cover - an attempt was just written
        raise ExplanationRefused("not_found")
    return view


async def attempts_used(session: AsyncSession, decision_id: UUID) -> int:
    total = await session.scalar(
        select(func.count())
        .select_from(ExplanationRow)
        .where(ExplanationRow.risk_decision_id == decision_id)
    )
    return int(total or 0)
