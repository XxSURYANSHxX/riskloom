from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    model_validator,
)

EventId = Annotated[str, StringConstraints(pattern=r"^evt_[0-9a-f]{32}$")]
MerchantId = Annotated[str, StringConstraints(pattern=r"^mrc_[0-9a-f]{32}$")]
CheckoutId = Annotated[str, StringConstraints(pattern=r"^chk_[0-9a-f]{32}$")]
CustomerToken = Annotated[str, StringConstraints(pattern=r"^cus_[0-9a-f]{32}$")]
DeviceToken = Annotated[str, StringConstraints(pattern=r"^dev_[0-9a-f]{32}$")]
NetworkToken = Annotated[str, StringConstraints(pattern=r"^net_[0-9a-f]{32}$")]
SessionToken = Annotated[str, StringConstraints(pattern=r"^ses_[0-9a-f]{32}$")]
PaymentInstrumentToken = Annotated[str, StringConstraints(pattern=r"^pmt_[0-9a-f]{32}$")]


class Outcome(StrEnum):
    AUTHORIZED = "authorized"
    FAILED = "failed"


class FailureCategory(StrEnum):
    AUTHENTICATION = "authentication"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INSTRUMENT_DECLINED = "instrument_declined"
    TEMPORARY_PROCESSING = "temporary_processing"
    UNKNOWN = "unknown"


class Channel(StrEnum):
    WEB = "web"
    MOBILE_WEB = "mobile_web"
    MOBILE_APP = "mobile_app"


class CheckoutAttemptEvent(BaseModel):
    """Strict model-visible synthetic checkout-attempt event."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    event_id: EventId
    merchant_id: MerchantId
    occurred_at: datetime
    checkout_id: CheckoutId
    customer_token: CustomerToken | None
    device_token: DeviceToken | None
    network_token: NetworkToken | None
    session_token: SessionToken
    payment_instrument_token: PaymentInstrumentToken
    amount_subunits: int = Field(ge=1, le=100_000_000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    outcome: Outcome
    failure_category: FailureCategory | None
    channel: Channel

    @model_validator(mode="after")
    def validate_outcome_failure_pair(self) -> Self:
        utc_offset = self.occurred_at.utcoffset()
        if utc_offset is None or utc_offset.total_seconds() != 0:
            raise ValueError("occurred_at must be timezone-aware UTC")
        if self.occurred_at.microsecond % 1_000:
            raise ValueError("occurred_at must use millisecond precision")
        if self.outcome is Outcome.AUTHORIZED and self.failure_category is not None:
            raise ValueError("authorized events must not have a failure category")
        if self.outcome is Outcome.FAILED and self.failure_category is None:
            raise ValueError("failed events must have a failure category")
        return self

    @field_serializer("occurred_at")
    def serialize_occurred_at(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1_000:03d}Z"
