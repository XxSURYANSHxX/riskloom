from pathlib import Path

import pytest

from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.generation import GeneratedRecord
from riskloom.simulation.validation import (
    DatasetValidationError,
    validate_event_privacy,
    validate_prohibited_keys,
    validate_typed_string,
)


@pytest.mark.parametrize(
    "field",
    [
        "card_number",
        "pan",
        "cvv",
        "cvc",
        "expiry",
        "email",
        "phone",
        "contact",
        "address",
        "billing_address",
        "vpa",
        "upi_id",
        "ip_address",
        "raw_payload",
    ],
)
def test_explicit_prohibited_key_denylist(field: str) -> None:
    with pytest.raises(DatasetValidationError, match="prohibited_field"):
        validate_prohibited_keys({"nested": [{field: "synthetic-prohibited-marker"}]})


@pytest.mark.parametrize(
    "field",
    ["Card-Number", "cardNumber", "billing address", "billingAddress", "RAW.PAYLOAD"],
)
def test_prohibited_keys_use_normalized_whole_key_matching(field: str) -> None:
    with pytest.raises(DatasetValidationError, match="prohibited_field"):
        validate_prohibited_keys({field: "synthetic-prohibited-marker"})
    validate_prohibited_keys({f"safe_{field}": "synthetic-marker"})


def test_payment_instrument_token_is_not_false_positive(
    tiny_records: list[GeneratedRecord],
) -> None:
    validate_prohibited_keys({"payment_instrument_token": "pmt_" + "a" * 32})
    for record in tiny_records:
        validate_event_privacy(record.event)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("channel", "person" + "@" + "example.invalid", "contact"),
        ("channel", "synthetic" + "@" + "handle", "contact"),
        ("channel", ".".join(["192", "0", "2", "1"]), "network"),
        ("channel", "4" * 16, "pan"),
        ("channel", "1" * 3, "cvv"),
    ],
)
def test_typed_sensitive_values_are_rejected(field: str, value: str, error: str) -> None:
    with pytest.raises(DatasetValidationError, match=error):
        validate_typed_string(field, value)


def test_safe_typed_values_are_not_misclassified() -> None:
    validate_typed_string("occurred_at", "2026-01-01T12:34:56.789Z")
    validate_typed_string("event_id", "evt_" + "1" * 32)
    validate_typed_string("channel", "mobile_web")


def test_no_name_or_faker_generation_surface() -> None:
    assert not any("name" in field for field in GeneratorConfig.model_fields)
    source = (Path(__file__).parents[3] / "src/riskloom/simulation/generation.py").read_text(
        encoding="utf-8"
    )
    assert "faker" not in source.casefold()
    assert "first_name" not in source
    assert "last_name" not in source
    assert "person_name" not in source
