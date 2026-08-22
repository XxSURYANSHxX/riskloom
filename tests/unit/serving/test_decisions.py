"""The locked-threshold decision rule, and the precision guarantee behind it."""

from decimal import Decimal

import pytest

from riskloom.services.preflight import _probability_decimal
from riskloom.serving.decisions import classify, decide, fail_safe
from riskloom.serving.schemas import DecisionAction, FailSafeReason, RiskDecision

# The real locked Day 4 threshold. Its float64 repr carries 19 significant decimals, one more than
# the Numeric(20, 18) audit column can hold.
LOCKED_THRESHOLD = 0.0033862949155182734
# The value the audit column rounds to. This is not an invented example: it is exactly the
# probability that a tie cluster of real scored rows carries, so a decision made from the rounded
# value would flip every row in that cluster.
ROUNDED_THRESHOLD = 0.003386294915518273


def test_threshold_boundary_is_inclusive_on_the_deny_side() -> None:
    assert classify(LOCKED_THRESHOLD, LOCKED_THRESHOLD) is RiskDecision.DENY
    assert classify(LOCKED_THRESHOLD * 2, LOCKED_THRESHOLD) is RiskDecision.DENY
    assert classify(0.0, LOCKED_THRESHOLD) is RiskDecision.ALLOW
    assert classify(ROUNDED_THRESHOLD, LOCKED_THRESHOLD) is RiskDecision.ALLOW


def test_decide_maps_risk_to_an_action_without_a_review_tier() -> None:
    allowed = decide(0.0, LOCKED_THRESHOLD)
    assert allowed.risk_decision is RiskDecision.ALLOW
    assert allowed.action is DecisionAction.ALLOW
    assert allowed.fail_safe_reason is None

    denied = decide(0.9, LOCKED_THRESHOLD)
    assert denied.risk_decision is RiskDecision.DENY
    assert denied.action is DecisionAction.DENY
    assert denied.fail_safe_reason is None

    # Risk is strictly two-way: no probability produces REVIEW.
    for probability in (0.0, 1e-9, ROUNDED_THRESHOLD, LOCKED_THRESHOLD, 0.5, 1.0):
        assert decide(probability, LOCKED_THRESHOLD).action is not DecisionAction.REVIEW


@pytest.mark.parametrize("reason", list(FailSafeReason))
def test_every_fail_safe_reason_routes_to_review(reason: FailSafeReason) -> None:
    decision = fail_safe(reason, RiskDecision.ALLOW)
    assert decision.action is DecisionAction.REVIEW
    assert decision.fail_safe_reason is reason
    # The underlying risk decision is preserved so the downgrade stays auditable.
    assert decision.risk_decision is RiskDecision.ALLOW


def test_audit_column_precision_would_change_the_decision_but_is_never_used() -> None:
    """Addition 2: the decision uses the full float64 value, never the audit round-trip.

    The Numeric(20, 18) column cannot represent the locked threshold exactly. A probability sitting
    on the rounded value is ALLOW under the true threshold and DENY under the rounded one, so the
    two are not interchangeable and the distinction is load-bearing, not cosmetic.
    """

    stored = _probability_decimal(LOCKED_THRESHOLD)
    assert stored == Decimal("0.003386294915518273")
    round_tripped = float(stored)

    assert round_tripped != LOCKED_THRESHOLD
    assert round_tripped == ROUNDED_THRESHOLD

    probability = ROUNDED_THRESHOLD
    assert classify(probability, LOCKED_THRESHOLD) is RiskDecision.ALLOW
    assert classify(probability, round_tripped) is RiskDecision.DENY


def test_serving_bundle_threshold_is_the_unrounded_model_value() -> None:
    """The value handed to `classify` comes straight from model.json, not from the database."""

    from riskloom.modeling.model import LockedModel  # noqa: PLC0415
    from riskloom.serving.model_host import ServingBundle  # noqa: PLC0415

    fields = set(ServingBundle.__dataclass_fields__)
    # There is no database session, connection or row on the bundle: it cannot read the column.
    assert fields == {"model", "feature_config", "feature_dataset_id"}
    assert "decision_threshold" in LockedModel.model_fields
