import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskloom.features.config import (
    FeatureConfig,
    FeatureConfigurationError,
    load_feature_config,
)
from riskloom.features.engine import FeatureEngine
from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES, FeatureRecord, FeatureVector
from riskloom.simulation.event_schema import CheckoutAttemptEvent

EventFactory = Callable[..., CheckoutAttemptEvent]


def test_feature_schema_has_exact_approved_family_counts() -> None:
    assert FEATURE_COUNT == 75
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    current = {
        "amount_subunits",
        "utc_hour",
        "utc_day_of_week",
        "channel_web",
        "channel_mobile_web",
        "channel_mobile_app",
        "customer_token_missing",
        "device_token_missing",
        "network_token_missing",
    }
    rates = {name for name in FEATURE_NAMES if name.endswith("_prior_failure_rate_300s_bp")}
    remaining = set(FEATURE_NAMES) - current - rates
    families = {
        "checkout": sum(name.startswith("checkout_") for name in remaining),
        "merchant": sum(name.startswith("merchant_") for name in remaining),
        "device": sum(name.startswith("device_") for name in remaining),
        "network": sum(name.startswith("network_") for name in remaining),
        "instrument": sum(name.startswith("instrument_") for name in remaining),
        "session": sum(name.startswith("session_") for name in remaining),
    }
    assert families == {
        "checkout": 3,
        "merchant": 15,
        "device": 12,
        "network": 15,
        "instrument": 10,
        "session": 6,
    }
    assert len(current) == 9
    assert len(rates) == 5
    assert len(current) + len(rates) + sum(families.values()) == FEATURE_COUNT


def test_feature_mapping_is_strict_and_coherent(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    record = FeatureEngine(feature_config).process(event_factory(1))
    values = record.features.model_dump()
    assert set(values) == set(FEATURE_NAMES)
    assert all(type(value) is int for value in values.values())

    with pytest.raises(ValidationError):
        FeatureVector.model_validate({**values, "unexpected": 1})
    with pytest.raises(ValidationError):
        FeatureVector.model_validate({**values, "amount_subunits": True})
    with pytest.raises(ValidationError):
        FeatureVector.model_validate({**values, "amount_subunits": 1.0})
    with pytest.raises(ValidationError):
        FeatureVector.model_validate({**values, "amount_subunits": "1"})
    with pytest.raises(ValidationError, match="channel"):
        FeatureVector.model_validate({**values, "channel_mobile_web": 1})
    with pytest.raises(ValidationError, match="history"):
        FeatureVector.model_validate({**values, "checkout_prior_attempt_count_3600s": 1})
    with pytest.raises(ValidationError, match="age"):
        FeatureVector.model_validate({**values, "checkout_previous_attempt_age_ms": 1})

    dumped = record.model_dump()
    with pytest.raises(ValidationError):
        FeatureRecord.model_validate({**dumped, "label": 1})
    with pytest.raises(ValidationError):
        FeatureRecord.model_validate(
            {**dumped, "features": {**values, "event_id": record.event_id}}
        )
    assert "event_id" not in values
    assert "occurred_at" not in values


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("utc_hour", 24),
        ("utc_day_of_week", 7),
        ("customer_token_missing", 2),
        ("merchant_prior_failure_rate_300s_bp", 10_001),
        ("merchant_prior_attempt_count_60s", -1),
        ("checkout_previous_attempt_age_ms", 3_600_000),
    ],
)
def test_feature_field_ranges_are_strict(
    field: str,
    value: int,
    feature_config: FeatureConfig,
    event_factory: EventFactory,
) -> None:
    values = FeatureEngine(feature_config).process(event_factory(1)).features.model_dump()
    with pytest.raises(ValidationError):
        FeatureVector.model_validate({**values, field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"merchant_prior_failure_count_60s": 1},
        {"merchant_distinct_devices_60s": 1},
        {"merchant_prior_attempt_count_60s": 1},
        {"merchant_prior_failure_rate_300s_bp": 1},
        {
            "device_token_missing": 1,
            "device_prior_attempt_count_60s": 1,
            "device_prior_attempt_count_300s": 1,
            "device_prior_attempt_count_3600s": 1,
        },
    ],
)
def test_feature_cross_field_coherence_is_enforced(
    changes: dict[str, int],
    feature_config: FeatureConfig,
    event_factory: EventFactory,
) -> None:
    values = FeatureEngine(feature_config).process(event_factory(1)).features.model_dump()
    with pytest.raises(ValidationError):
        FeatureVector.model_validate({**values, **changes})


def test_locked_configuration_loads_and_rejects_changes(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    path = tmp_path / "feature-config.json"
    path.write_text(json.dumps(feature_config.model_dump(mode="json")), encoding="utf-8")
    assert load_feature_config(path) == feature_config

    changed = feature_config.model_dump(mode="json")
    changed["rolling_windows_seconds"] = [60, 301, 3600]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FeatureConfigurationError, match="invalid"):
        load_feature_config(path)

    path.write_bytes(b"x" * 65_537)
    with pytest.raises(FeatureConfigurationError, match="oversized"):
        load_feature_config(path)
    with pytest.raises(FeatureConfigurationError, match="invalid"):
        load_feature_config(tmp_path / "missing.json")
