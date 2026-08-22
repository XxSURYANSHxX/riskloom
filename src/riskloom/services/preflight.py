"""Live preflight orchestration: claim, score, act, finalise.

Ordering is deliberate and load-bearing. The idempotency claim happens *before* the engine is
touched, so a retried request can never advance rolling state twice or create a second order. Only
after the claim succeeds is the event scored, and only then is an action attempted.
"""

from dataclasses import dataclass
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.db.models import ReviewItem
from riskloom.db.models import RiskDecision as RiskDecisionRow
from riskloom.features.config import FEATURE_ENGINE_VERSION, FEATURE_SCHEMA_VERSION
from riskloom.integrations.razorpay.client import RazorpayOrdersClient, RazorpayOrdersError
from riskloom.serving.decisions import Decision, decide, fail_safe
from riskloom.serving.engine_host import OnlineFeatureEngine
from riskloom.serving.model_host import ServingBundle
from riskloom.serving.schemas import (
    CheckoutPreflightRequest,
    DecisionAction,
    FailSafeReason,
    PreflightDecisionResponse,
    RiskDecision,
)

logger = structlog.get_logger("riskloom.preflight")


class PreflightPendingError(Exception):
    """A prior attempt at this event claimed the ledger row and never finalised it."""


@dataclass(slots=True)
class OrderBudget:
    """Process-level cap on Razorpay order-creation *attempts*.

    This makes the project's "a few, not hundreds" constraint structural rather than a promise.
    Past the cap an ALLOW fail-safes to REVIEW instead of attempting an order.

    The budget is spent by attempts, not by successes: ``take`` is called before the upstream
    request, so an attempt that Razorpay then rejects still consumes one unit. That is deliberate
    and is the safer direction -- what needs bounding is outbound calls to a payment provider, and
    counting only successes would let an endlessly failing path retry without limit. It does mean
    the counter can exceed the number of orders that actually exist.
    """

    limit: int
    attempted: int = 0

    def take(self) -> bool:
        """Reserve one attempt. ``False`` once the cap is reached."""

        if self.attempted >= self.limit:
            return False
        self.attempted += 1
        return True


def _probability_decimal(probability: float | None) -> Decimal | None:
    """Audit representation only; never read back to make a decision."""

    if probability is None:
        return None
    return Decimal(repr(probability)).quantize(Decimal("1.000000000000000000"))


def _response(
    row: RiskDecisionRow, *, duplicate: bool, decision_threshold: float
) -> PreflightDecisionResponse:
    """Build the response.

    ``decision_threshold`` is passed in from the locked model rather than read back from the
    Numeric(20, 18) column, which rounds it. The decision itself already used the full float64
    value; reporting the rounded one would misdescribe the rule that was actually applied.
    """

    return PreflightDecisionResponse(
        decision_id=row.id,
        event_id=row.event_id,
        action=DecisionAction(row.action) if row.action else DecisionAction.REVIEW,
        risk_decision=RiskDecision(row.risk_decision) if row.risk_decision else None,
        calibrated_probability=(
            float(row.calibrated_probability) if row.calibrated_probability is not None else None
        ),
        decision_threshold=decision_threshold,
        model_id=row.model_id,
        fail_safe_reason=(FailSafeReason(row.fail_safe_reason) if row.fail_safe_reason else None),
        razorpay_order_id=row.razorpay_order_id,
        evaluated_at=row.occurred_at or row.created_at,
        duplicate=duplicate,
    )


async def _claim(
    session: AsyncSession, request: CheckoutPreflightRequest, bundle: ServingBundle
) -> RiskDecisionRow | None:
    """Reserve the event id, reusing the Day 1 webhook idempotency pattern.

    ``None`` means another request already claimed this event id.
    """

    async with session.begin():
        claimed = await session.scalar(
            insert(RiskDecisionRow)
            .values(
                event_id=request.event_id,
                merchant_id=request.merchant_id,
                checkout_id=request.checkout_id,
                session_token=request.session_token,
                payment_instrument_token=request.payment_instrument_token,
                customer_token=request.customer_token,
                device_token=request.device_token,
                network_token=request.network_token,
                amount_subunits=request.amount_subunits,
                currency=request.currency,
                channel=request.channel.value,
                decision_threshold=_probability_decimal(bundle.decision_threshold),
                model_id=bundle.model.model_id,
                feature_dataset_id=bundle.feature_dataset_id,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_engine_version=FEATURE_ENGINE_VERSION,
                status="pending",
            )
            .on_conflict_do_nothing(index_elements=[RiskDecisionRow.event_id])
            .returning(RiskDecisionRow.id)
        )
        if claimed is None:
            return None
        # Load inside this transaction. Reading after commit would autobegin another one and the
        # later explicit begin() for finalisation would raise.
        return await session.get(RiskDecisionRow, claimed)


async def _existing(session: AsyncSession, event_id: str) -> RiskDecisionRow | None:
    result = await session.scalar(
        select(RiskDecisionRow).where(RiskDecisionRow.event_id == event_id)
    )
    return result


async def _act(
    decision: Decision,
    request: CheckoutPreflightRequest,
    orders: RazorpayOrdersClient,
    budget: OrderBudget,
) -> tuple[Decision, str | None]:
    """Carry out the decided action. Returns the possibly fail-safed decision and any order id."""

    if decision.action is not DecisionAction.ALLOW:
        return decision, None

    if not budget.take():
        return fail_safe(FailSafeReason.ORDER_BUDGET_EXHAUSTED, decision.risk_decision), None

    # Razorpay caps the receipt field at 40 characters; "rl_" plus the 32 hex characters of the
    # event id is 35, safely inside that limit and unique per event.
    receipt = f"rl_{request.event_id[4:]}"
    try:
        order = await orders.create_order(
            amount=request.amount_subunits, currency=request.currency, receipt=receipt
        )
    except RazorpayOrdersError:
        logger.warning("preflight_order_creation_failed", event_id=request.event_id)
        return fail_safe(FailSafeReason.ORDER_CREATION_FAILED, decision.risk_decision), None
    return decision, order.id


async def evaluate_preflight(
    session: AsyncSession,
    request: CheckoutPreflightRequest,
    *,
    engine: OnlineFeatureEngine,
    bundle: ServingBundle,
    orders: RazorpayOrdersClient,
    budget: OrderBudget,
) -> PreflightDecisionResponse:
    row = await _claim(session, request, bundle)
    if row is None:
        existing = await _existing(session, request.event_id)
        if existing is None or existing.status != "final":
            raise PreflightPendingError(request.event_id)
        return _response(existing, duplicate=True, decision_threshold=bundle.decision_threshold)

    probability: float | None = None
    try:
        scored = await engine.process(request)
    except Exception:
        logger.warning("preflight_feature_computation_failed", event_id=request.event_id)
        decision = fail_safe(FailSafeReason.FEATURE_COMPUTATION_FAILED)
        occurred_at = None
    else:
        occurred_at = scored.occurred_at
        try:
            probability = bundle.probability(scored.record)
        except Exception:
            logger.warning("preflight_scoring_failed", event_id=request.event_id)
            decision = fail_safe(FailSafeReason.SCORING_FAILED)
        else:
            # The full float64 threshold straight from model.json, never the audit column.
            decision = decide(probability, bundle.decision_threshold)

    order_id: str | None = None
    if occurred_at is not None:
        decision, order_id = await _act(decision, request, orders, budget)

    async with session.begin():
        row.occurred_at = occurred_at or row.created_at
        row.calibrated_probability = _probability_decimal(probability)
        row.risk_decision = decision.risk_decision.value if decision.risk_decision else None
        row.action = decision.action.value
        row.fail_safe_reason = (
            decision.fail_safe_reason.value if decision.fail_safe_reason else None
        )
        row.razorpay_order_id = order_id
        row.status = "final"
        session.add(row)
        if decision.action is DecisionAction.REVIEW:
            await session.execute(
                insert(ReviewItem)
                .values(
                    risk_decision_id=row.id,
                    merchant_id=request.merchant_id,
                    checkout_id=request.checkout_id,
                    status="pending",
                )
                .on_conflict_do_nothing(index_elements=[ReviewItem.risk_decision_id])
            )

    logger.info(
        "preflight_decided",
        event_id=request.event_id,
        action=decision.action.value,
        risk_decision=decision.risk_decision.value if decision.risk_decision else None,
        fail_safe_reason=(decision.fail_safe_reason.value if decision.fail_safe_reason else None),
        order_created=order_id is not None,
    )
    return _response(row, duplicate=False, decision_threshold=bundle.decision_threshold)
