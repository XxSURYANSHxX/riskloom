"""Explanation evidence must describe what happened *before* the decision it explains.

The defect these tests lock out was not a wrong comparison. `entity_context_for` filtered on token
equality alone, so an entity whose only appearance in the ledger was the decision being explained
still reported `denied_count = 1` -- counting itself -- and `factors._denied`'s `denied_count > 0`
predicate duly passed. A rotated card-testing instrument, which by design appears exactly once,
therefore supported "prior denials on this instrument" when there were none.

The class of bug is a self-referential read against an already-committed row, so the tests below are
written as an invariant over the pipeline rather than as a fixture for the one factor that was
caught: no count-based factor may be entailed by evidence that requires counting the
decision itself.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from riskloom.db.models import RiskDecision
from riskloom.explanations import factors
from riskloom.explanations.schemas import FactorCode
from riskloom.services.dashboard import SelfInclusion, entity_context_for
from riskloom.services.explanations import build_input

pytestmark = pytest.mark.integration

THRESHOLD = Decimal("0.003386294915518273")

# Every factor whose wording claims history. These are the ones that must never be entailed by a
# count that includes the decision under explanation.
HISTORY_FACTORS = frozenset(
    {
        FactorCode.PRIOR_DENIALS_ON_DEVICE,
        FactorCode.PRIOR_DENIALS_ON_NETWORK,
        FactorCode.PRIOR_DENIALS_ON_INSTRUMENT,
        FactorCode.DEVICE_REUSE,
        FactorCode.NETWORK_REUSE,
        FactorCode.INSTRUMENT_REUSE,
        FactorCode.MERCHANT_VOLUME,
        FactorCode.RAPID_SUCCESSION,
    }
)


def decision(
    index: int,
    *,
    action: str,
    device: str,
    network: str,
    instrument: str,
    merchant: str,
    age_seconds: int = 0,
) -> RiskDecision:
    now = datetime.now(UTC) - timedelta(seconds=age_seconds)
    return RiskDecision(
        id=uuid4(),
        event_id=f"evt_{index:032x}",
        merchant_id=merchant,
        checkout_id=f"chk_{index:032x}",
        session_token=f"ses_{index:032x}",
        payment_instrument_token=instrument,
        customer_token=None,
        device_token=device,
        network_token=network,
        amount_subunits=25_000,
        currency="INR",
        channel="web",
        occurred_at=now,
        calibrated_probability=Decimal("0.007053679692244301"),
        decision_threshold=THRESHOLD,
        risk_decision="deny" if action == "deny" else "allow",
        action=action,
        fail_safe_reason=None,
        model_id="d" * 64,
        feature_dataset_id="e" * 64,
        feature_schema_version="1.0.0",
        feature_engine_version="1.0.0",
        razorpay_order_id=None,
        status="final",
    )


def seed(url: str, rows: list[RiskDecision]) -> None:
    async def run() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "TRUNCATE risk_decision_explanations, review_items, risk_decisions "
                        "RESTART IDENTITY CASCADE"
                    )
                )
            async with AsyncSession(engine, expire_on_commit=False) as session, session.begin():
                for row in rows:
                    session.add(row)
        finally:
            await engine.dispose()

    asyncio.run(run())


def evidence(url: str, decision_id: UUID, mode: SelfInclusion) -> Any:
    """The exact aggregate the explanation pipeline would send, for one decision."""

    async def run() -> Any:
        engine = create_async_engine(url)
        try:
            async with AsyncSession(engine) as session:
                row = await session.scalar(
                    select(RiskDecision).where(RiskDecision.id == decision_id)
                )
                assert row is not None
                context = await entity_context_for(session, row, mode)
                return build_input(row, context)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def supported(payload: Any) -> set[FactorCode]:
    return set(factors.available(payload))


# --------------------------------------------------------------------------- the invariant


def test_no_history_factor_survives_removing_the_decision_itself(
    migrated_database_url: str,
) -> None:
    """The invariant, stated directly: an entity seen exactly once supports no history factor.

    This is the whole class of bug, not the one instance of it. Every entity on this decision --
    device, network, instrument and merchant -- appears exactly once, so there is no prior history
    of any kind and nothing that counts history may be entailed.
    """

    only = decision(
        1,
        action="deny",
        device="dev_" + "1" * 32,
        network="net_" + "1" * 32,
        instrument="pmt_" + "1" * 32,
        merchant="mrc_" + "1" * 32,
    )
    seed(migrated_database_url, [only])

    payload = evidence(migrated_database_url, only.id, SelfInclusion.EXCLUDE_SELF)
    assert not supported(payload) & HISTORY_FACTORS, supported(payload)

    for aggregate in payload.context:
        assert aggregate.decision_count == 0, aggregate.kind
        assert aggregate.denied_count == 0, aggregate.kind
        assert aggregate.review_count == 0, aggregate.kind
        assert aggregate.span_seconds in (0, None), aggregate.kind


def test_the_unfixed_query_would_have_entailed_a_false_factor(
    migrated_database_url: str,
) -> None:
    """The defect, reproduced through the same primitive, so the fix is shown to be load-bearing.

    Including the decision itself makes a single-appearance instrument report one denial, and
    ``prior_denials_on_instrument`` becomes entailed. Asserting the broken behaviour here means the
    corrected assertion above cannot pass vacuously.
    """

    only = decision(
        2,
        action="deny",
        device="dev_" + "2" * 32,
        network="net_" + "2" * 32,
        instrument="pmt_" + "2" * 32,
        merchant="mrc_" + "2" * 32,
    )
    seed(migrated_database_url, [only])

    contaminated = supported(evidence(migrated_database_url, only.id, SelfInclusion.INCLUDE_SELF))
    corrected = supported(evidence(migrated_database_url, only.id, SelfInclusion.EXCLUDE_SELF))

    assert FactorCode.PRIOR_DENIALS_ON_INSTRUMENT in contaminated
    assert FactorCode.PRIOR_DENIALS_ON_INSTRUMENT not in corrected
    assert corrected < contaminated


def test_a_first_ever_deny_still_supports_what_is_genuinely_true(
    migrated_database_url: str,
) -> None:
    """Closing the leak must not leave the explanation with nothing to say.

    The first denial a system ever makes has no history behind it, but the probability did cross
    the locked threshold, and that remains supportable because it is a property of the decision
    rather than of its past.
    """

    first = decision(
        3,
        action="deny",
        device="dev_" + "3" * 32,
        network="net_" + "3" * 32,
        instrument="pmt_" + "3" * 32,
        merchant="mrc_" + "3" * 32,
    )
    seed(migrated_database_url, [first])

    payload = evidence(migrated_database_url, first.id, SelfInclusion.EXCLUDE_SELF)
    assert supported(payload) == {FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD}
    assert payload.probability_exceeds_threshold is True


# --------------------------------------------------------------------------- the masked case


def test_real_prior_history_is_counted_correctly_and_is_exactly_one_less(
    migrated_database_url: str,
) -> None:
    """Where the defect was masked, the fix must subtract exactly the decision itself.

    A device with genuine history still supports its factors; the count simply loses the current
    row. Off-by-one in either direction fails here.
    """

    device = "dev_" + "4" * 32
    network = "net_" + "4" * 32
    merchant = "mrc_" + "4" * 32
    rows = [
        decision(
            10 + i,
            action="deny" if i % 2 == 0 else "allow",
            device=device,
            network=network,
            instrument=f"pmt_{10 + i:032x}",
            merchant=merchant,
            age_seconds=120 - i * 10,
        )
        for i in range(6)
    ]
    subject = rows[-1]
    seed(migrated_database_url, rows)

    included = evidence(migrated_database_url, subject.id, SelfInclusion.INCLUDE_SELF)
    excluded = evidence(migrated_database_url, subject.id, SelfInclusion.EXCLUDE_SELF)

    def by_kind(payload: Any, kind: str) -> Any:
        return next(a for a in payload.context if a.kind == kind)

    for kind in ("device", "network", "merchant"):
        before, after = by_kind(included, kind), by_kind(excluded, kind)
        assert after.decision_count == before.decision_count - 1, kind
        # The subject is an allow (index 5 is odd), so the denial count is untouched by removing it.
        assert after.denied_count == before.denied_count, kind

    # Reuse is real here and must survive the correction.
    assert FactorCode.DEVICE_REUSE in supported(excluded)
    assert FactorCode.PRIOR_DENIALS_ON_DEVICE in supported(excluded)


def test_removing_a_denial_lowers_the_denial_count_by_exactly_one(
    migrated_database_url: str,
) -> None:
    """The complementary case: when the subject *is* a denial, excluding it must show."""

    device = "dev_" + "5" * 32
    rows = [
        decision(
            20 + i,
            action="deny",
            device=device,
            network="net_" + "5" * 32,
            instrument=f"pmt_{20 + i:032x}",
            merchant="mrc_" + "5" * 32,
            age_seconds=60 - i * 10,
        )
        for i in range(3)
    ]
    subject = rows[-1]
    seed(migrated_database_url, rows)

    included = evidence(migrated_database_url, subject.id, SelfInclusion.INCLUDE_SELF)
    excluded = evidence(migrated_database_url, subject.id, SelfInclusion.EXCLUDE_SELF)
    before = next(a for a in included.context if a.kind == "device")
    after = next(a for a in excluded.context if a.kind == "device")

    assert before.denied_count == 3
    assert after.denied_count == 2
    # Two genuine prior denials remain, so the factor is still correctly entailed.
    assert FactorCode.PRIOR_DENIALS_ON_DEVICE in supported(excluded)


# --------------------------------------------------------------------------- the boundary itself


def test_the_inclusion_choice_cannot_be_skipped() -> None:
    """The argument has no default, so no future call site can inherit the old behaviour."""

    import inspect

    signature = inspect.signature(entity_context_for)
    parameter = signature.parameters["self_inclusion"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.annotation is SelfInclusion


def test_every_call_site_states_its_choice() -> None:
    """Both callers name a member explicitly; neither passes a computed or defaulted value."""

    import re
    from pathlib import Path

    calls = []
    for path in Path("src/riskloom").rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "entity_context_for(" in line and "def " not in line:
                calls.append((path.name, line.strip()))

    assert len(calls) == 2, calls
    for name, line in calls:
        assert re.search(r"SelfInclusion\.(INCLUDE_SELF|EXCLUDE_SELF)", line), (name, line)


def test_the_explanation_path_uses_exclude_self() -> None:
    """The one call that must never change, asserted by name rather than by behaviour alone."""

    from pathlib import Path

    source = Path("src/riskloom/services/explanations.py").read_text(encoding="utf-8")
    assert "entity_context_for(session, row, SelfInclusion.EXCLUDE_SELF)" in source
