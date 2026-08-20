from collections.abc import Callable, Mapping
from typing import Any

from riskloom.integrations.razorpay.schemas import (
    CURRENCY_PATTERN,
    SAFE_IDENTIFIER_PATTERN,
    SAFE_TOKEN_PATTERN,
)

REDACTED = "[REDACTED]"

_EVENT_FIELDS: dict[str, Callable[[Any], bool]] = {
    "entity": lambda value: value == "event",
    "event": lambda value: _safe_string(value, SAFE_TOKEN_PATTERN, 100),
    "created_at": lambda value: _safe_timestamp(value),
    "account_id": lambda value: _safe_string(value, SAFE_IDENTIFIER_PATTERN, 255),
}

_PAYMENT_FIELDS: dict[str, Callable[[Any], bool]] = {
    "id": lambda value: _safe_string(value, SAFE_IDENTIFIER_PATTERN, 255),
    "entity": lambda value: value == "payment",
    "amount": lambda value: _safe_non_negative_integer(value),
    "amount_refunded": lambda value: _safe_non_negative_integer(value),
    "currency": lambda value: _safe_string(value, CURRENCY_PATTERN, 3),
    "status": lambda value: _safe_string(value, SAFE_TOKEN_PATTERN, 50),
    "order_id": lambda value: value is None or _safe_string(value, SAFE_IDENTIFIER_PATTERN, 255),
    "invoice_id": lambda value: value is None or _safe_string(value, SAFE_IDENTIFIER_PATTERN, 255),
    "international": lambda value: isinstance(value, bool),
    "method": lambda value: value is None or _safe_string(value, SAFE_TOKEN_PATTERN, 50),
    "captured": lambda value: isinstance(value, bool),
    "created_at": lambda value: _safe_timestamp(value),
    "refund_status": lambda value: value is None or _safe_string(value, SAFE_TOKEN_PATTERN, 50),
    "error_code": lambda value: value is None or _safe_string(value, SAFE_TOKEN_PATTERN, 100),
    "error_source": lambda value: value is None or _safe_string(value, SAFE_TOKEN_PATTERN, 100),
    "error_step": lambda value: value is None or _safe_string(value, SAFE_TOKEN_PATTERN, 100),
    "error_reason": lambda value: value is None or _safe_string(value, SAFE_TOKEN_PATTERN, 100),
}

_ORDER_FIELDS: dict[str, Callable[[Any], bool]] = {
    "id": lambda value: _safe_string(value, SAFE_IDENTIFIER_PATTERN, 255),
    "entity": lambda value: value == "order",
    "amount": lambda value: _safe_non_negative_integer(value),
    "amount_paid": lambda value: _safe_non_negative_integer(value),
    "amount_due": lambda value: _safe_non_negative_integer(value),
    "currency": lambda value: _safe_string(value, CURRENCY_PATTERN, 3),
    "status": lambda value: _safe_string(value, SAFE_TOKEN_PATTERN, 50),
    "attempts": lambda value: _safe_non_negative_integer(value),
    "created_at": lambda value: _safe_timestamp(value),
}

_PAYMENT_REDACTIONS = {
    "acquirer_data",
    "card",
    "contact",
    "description",
    "email",
    "ip",
    "ip_address",
    "notes",
    "phone",
    "token_id",
    "vpa",
}
_ORDER_REDACTIONS = {"notes"}


def _safe_string(value: Any, pattern: str, maximum_length: int) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum_length:
        return False
    import re

    return re.fullmatch(pattern, value) is not None


def _safe_non_negative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _safe_timestamp(value: Any) -> bool:
    return _safe_non_negative_integer(value) and value <= 4_102_444_800


def _copy_allowlisted(
    source: Mapping[str, Any],
    allowlist: Mapping[str, Callable[[Any], bool]],
    redacted_fields: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, validator in allowlist.items():
        if key in source:
            value = source[key]
            result[key] = value if validator(value) else REDACTED
    for key in redacted_fields:
        if key in source:
            result[key] = REDACTED
    return result


def sanitize_webhook_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    """Create a non-recursive-by-default, explicit audit projection.

    Nested values are only visited along documented payment/order entity paths. Unknown keys and
    free-form objects are excluded, while known sensitive keys are represented by a constant marker.
    """

    result = _copy_allowlisted(source, _EVENT_FIELDS, set())

    contains = source.get("contains")
    if isinstance(contains, list) and len(contains) <= 20:
        safe_contains = [value for value in contains if _safe_string(value, SAFE_TOKEN_PATTERN, 50)]
        if len(safe_contains) == len(contains):
            result["contains"] = safe_contains

    payload = source.get("payload")
    if not isinstance(payload, Mapping):
        return result

    projected_payload: dict[str, Any] = {}
    for entity_name, allowlist, redactions in (
        ("payment", _PAYMENT_FIELDS, _PAYMENT_REDACTIONS),
        ("order", _ORDER_FIELDS, _ORDER_REDACTIONS),
    ):
        wrapper = payload.get(entity_name)
        if not isinstance(wrapper, Mapping):
            continue
        entity = wrapper.get("entity")
        if not isinstance(entity, Mapping):
            continue
        projected_payload[entity_name] = {
            "entity": _copy_allowlisted(entity, allowlist, redactions)
        }

    if projected_payload:
        result["payload"] = projected_payload
    return result
