from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SAFE_IDENTIFIER_PATTERN = r"^[A-Za-z0-9_-]+$"
SAFE_TOKEN_PATTERN = r"^[A-Za-z0-9_.-]+$"
CURRENCY_PATTERN = r"^[A-Z]{3}$"


class WebhookEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    entity: Literal["event"]
    event: str = Field(min_length=1, max_length=100, pattern=SAFE_TOKEN_PATTERN)
    created_at: int = Field(ge=0, le=4_102_444_800)
    payload: dict[str, Any]
    contains: list[str] = Field(default_factory=list, max_length=20)
    account_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=SAFE_IDENTIFIER_PATTERN,
    )


class PaymentEntity(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str = Field(min_length=1, max_length=255, pattern=SAFE_IDENTIFIER_PATTERN)
    entity: Literal["payment"]
    order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=SAFE_IDENTIFIER_PATTERN,
    )
    status: str = Field(min_length=1, max_length=50, pattern=SAFE_TOKEN_PATTERN)
    amount: int = Field(ge=0)
    currency: str = Field(pattern=CURRENCY_PATTERN)
    method: str | None = Field(default=None, max_length=50, pattern=SAFE_TOKEN_PATTERN)
    created_at: int = Field(ge=0, le=4_102_444_800)
    error_code: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_.-]*$")
    error_source: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_.-]*$")
    error_step: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_.-]*$")
    error_reason: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_.-]*$")


class CreateOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    amount: int = Field(ge=1)
    currency: str = Field(pattern=CURRENCY_PATTERN)
    receipt: str = Field(min_length=1, max_length=40, pattern=r"^[\x20-\x7E]+$")


class RazorpayOrder(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str = Field(min_length=1, max_length=255, pattern=SAFE_IDENTIFIER_PATTERN)
    entity: Literal["order"]
    amount: int = Field(ge=0)
    amount_paid: int = Field(ge=0)
    amount_due: int = Field(ge=0)
    currency: str = Field(pattern=CURRENCY_PATTERN)
    receipt: str = Field(max_length=40)
    status: Literal["created", "attempted", "paid"]
    attempts: int = Field(ge=0)
    created_at: int = Field(ge=0, le=4_102_444_800)
