"""The live decision rule.

Risk is strictly two-way and comes from the locked Day 4 model's single ``decision_threshold``:
below it allows, at or above it denies. There is no second risk threshold anywhere in this module
or reachable from it, and ``riskloom.policy`` is never imported -- neither Gate C1 band takes part
in any real decision.

REVIEW is not a risk band. It is an operational fail-safe tier, reached only when a decision could
not be safely completed.
"""

from dataclasses import dataclass

from riskloom.serving.schemas import DecisionAction, FailSafeReason, RiskDecision


@dataclass(frozen=True, slots=True)
class Decision:
    risk_decision: RiskDecision | None
    action: DecisionAction
    fail_safe_reason: FailSafeReason | None


def classify(probability: float, decision_threshold: float) -> RiskDecision:
    """Apply the locked threshold.

    Both arguments must be the full float64 values: ``probability`` as returned by portable
    inference, and ``decision_threshold`` as loaded from model.json. Neither may be a value that
    has round-tripped through the Numeric(20, 18) audit column, which exists for storage and audit
    only and can round differently at the eighteenth decimal place.
    """

    if probability >= decision_threshold:
        return RiskDecision.DENY
    return RiskDecision.ALLOW


def decide(probability: float, decision_threshold: float) -> Decision:
    """The risk decision before any action is attempted."""

    risk_decision = classify(probability, decision_threshold)
    action = DecisionAction.DENY if risk_decision is RiskDecision.DENY else DecisionAction.ALLOW
    return Decision(risk_decision=risk_decision, action=action, fail_safe_reason=None)


def fail_safe(reason: FailSafeReason, risk_decision: RiskDecision | None = None) -> Decision:
    """Route to human review because the decision could not be safely completed."""

    return Decision(
        risk_decision=risk_decision,
        action=DecisionAction.REVIEW,
        fail_safe_reason=reason,
    )
