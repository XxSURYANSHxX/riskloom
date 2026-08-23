"""Request and response contracts for live checkout-preflight scoring.

The request reuses the Day 2 pseudonymous token types verbatim, so personally identifying data is
structurally unsubmittable: there is no field that would accept an email address, phone number,
card number, cardholder name, VPA or network address, and ``extra="forbid"`` rejects any attempt
to add one.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from riskloom.simulation.event_schema import (
    Channel,
    CheckoutId,
    CustomerToken,
    DeviceToken,
    EventId,
    MerchantId,
    NetworkToken,
    PaymentInstrumentToken,
    SessionToken,
)


class RiskDecision(StrEnum):
    """What the locked model's single threshold concluded."""

    ALLOW = "allow"
    DENY = "deny"


class DecisionAction(StrEnum):
    """What the service actually did, which can differ from the risk decision."""

    ALLOW = "allow"
    REVIEW = "review"
    DENY = "deny"


class FailSafeReason(StrEnum):
    """Why an action was routed to human review instead of being completed.

    These are operational conditions, never risk bands. REVIEW exists in this gate solely because
    a decision could not be safely completed, so no second risk threshold is implied by any of
    them.
    """

    FEATURE_COMPUTATION_FAILED = "feature_computation_failed"
    SCORING_FAILED = "scoring_failed"
    ORDER_CREATION_FAILED = "order_creation_failed"
    ORDER_BUDGET_EXHAUSTED = "order_budget_exhausted"


class CheckoutPreflightRequest(BaseModel):
    """A single live checkout attempt awaiting a decision.

    ``occurred_at`` is deliberately absent: the server assigns a strictly increasing timestamp
    under the engine lock. That keeps the feature engine's monotonic ordering guarantee unbreakable
    and stops a caller replaying stale timestamps to poison rolling state.

    ``outcome`` and ``failure_category`` are deliberately absent too: at preflight the attempt has
    not happened, so its outcome does not exist yet.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    event_id: EventId
    merchant_id: MerchantId
    checkout_id: CheckoutId
    customer_token: CustomerToken | None = None
    device_token: DeviceToken | None = None
    network_token: NetworkToken | None = None
    session_token: SessionToken
    payment_instrument_token: PaymentInstrumentToken
    # No lower bound beyond the schema's own positivity requirement. An amount below Razorpay's
    # documented INR minimum of 100 paise is passed through unmodified so that upstream rejection
    # is genuine rather than fabricated locally.
    amount_subunits: int = Field(ge=1, le=100_000_000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    channel: Channel


class PreflightDecisionResponse(BaseModel):
    """The decision returned to the caller.

    Deliberately excluded: the computed feature vector, engine state or diagnostics, any other
    merchant's or checkout's data, upstream Razorpay response bodies, and anything credential
    bearing. Explanations are generated separately, on their own endpoint, after the fact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: UUID
    event_id: str
    action: DecisionAction
    risk_decision: RiskDecision | None
    calibrated_probability: float | None
    decision_threshold: float
    model_id: str
    fail_safe_reason: FailSafeReason | None
    razorpay_order_id: str | None
    evaluated_at: datetime
    duplicate: bool
