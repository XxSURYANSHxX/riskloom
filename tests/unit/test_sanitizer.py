import json
from typing import Any

from riskloom.integrations.razorpay.sanitizer import REDACTED, sanitize_webhook_payload


def test_sanitizer_is_allowlist_only_and_recursively_redacts(
    synthetic_event: dict[str, Any],
) -> None:
    projection = sanitize_webhook_payload(synthetic_event)
    serialized = json.dumps(projection, sort_keys=True)

    assert projection["event"] == "payment.failed"
    payment = projection["payload"]["payment"]["entity"]
    assert payment["id"] == "pay_synthetic"
    assert payment["email"] == REDACTED
    assert payment["card"] == REDACTED
    for forbidden in (
        "person@example.invalid",
        "+10000000000",
        "192.0.2.1",
        "synthetic-pan-marker",
        "synthetic-cvv-marker",
        "Synthetic Cardholder",
        "synthetic note",
        "must not persist",
    ):
        assert forbidden not in serialized


def test_sanitizer_omits_invalid_allowlisted_values() -> None:
    projection = sanitize_webhook_payload(
        {
            "entity": "event",
            "event": "invalid event with spaces",
            "created_at": -1,
            "contains": ["payment", "not safe"],
            "payload": "not-an-object",
        }
    )

    assert projection == {
        "entity": "event",
        "event": REDACTED,
        "created_at": REDACTED,
    }


def test_sanitizer_projects_safe_order_fields_only() -> None:
    projection = sanitize_webhook_payload(
        {
            "entity": "event",
            "event": "order.paid",
            "created_at": 1_700_000_000,
            "payload": {
                "order": {
                    "entity": {
                        "id": "order_synthetic",
                        "entity": "order",
                        "amount": 100,
                        "amount_paid": 100,
                        "amount_due": 0,
                        "currency": "INR",
                        "status": "paid",
                        "attempts": 1,
                        "created_at": 1_699_999_000,
                        "notes": {"unsafe": "synthetic"},
                        "receipt": "free form excluded",
                    }
                }
            },
        }
    )

    order = projection["payload"]["order"]["entity"]
    assert order["notes"] == REDACTED
    assert "receipt" not in order
