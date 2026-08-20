from collections.abc import Callable
from datetime import UTC

import pytest

from riskloom.features.artifacts import canonical_json_bytes
from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine, FeatureEngineInputError
from riskloom.simulation.event_schema import CheckoutAttemptEvent, Outcome

EventFactory = Callable[..., CheckoutAttemptEvent]


def test_compute_before_update_and_current_outcome_invariance(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    authorized_engine = FeatureEngine(feature_config)
    failed_engine = FeatureEngine(feature_config)
    authorized = event_factory(1, outcome=Outcome.AUTHORIZED)
    failed = event_factory(1, outcome=Outcome.FAILED)

    authorized_record = authorized_engine.process(authorized)
    failed_record = failed_engine.process(failed)
    assert authorized_record == failed_record
    assert canonical_json_bytes(authorized_record.model_dump(mode="json")) == canonical_json_bytes(
        failed_record.model_dump(mode="json")
    )
    assert authorized_record.features.merchant_prior_attempt_count_60s == 0
    assert authorized_record.features.merchant_prior_failure_count_60s == 0

    next_event = event_factory(2, seconds=1)
    authorized_next = authorized_engine.process(next_event)
    failed_next = failed_engine.process(next_event)
    assert authorized_next.features.merchant_prior_attempt_count_60s == 1
    assert failed_next.features.merchant_prior_attempt_count_60s == 1
    assert authorized_next.features.merchant_prior_failure_count_60s == 0
    assert failed_next.features.merchant_prior_failure_count_60s == 1
    assert authorized_next.features.merchant_prior_failure_rate_300s_bp == 0
    assert failed_next.features.merchant_prior_failure_rate_300s_bp == 10_000
    authorized_values = authorized_next.features.model_dump()
    failed_values = failed_next.features.model_dump()
    changed = {name for name in authorized_values if authorized_values[name] != failed_values[name]}
    assert changed == {
        "merchant_prior_failure_count_60s",
        "merchant_prior_failure_count_300s",
        "merchant_prior_failure_count_3600s",
        "device_prior_failure_count_60s",
        "device_prior_failure_count_300s",
        "device_prior_failure_count_3600s",
        "network_prior_failure_count_60s",
        "network_prior_failure_count_300s",
        "network_prior_failure_count_3600s",
        "instrument_prior_failure_count_300s",
        "instrument_prior_failure_count_3600s",
        "session_prior_failure_count_60s",
        "session_prior_failure_count_300s",
        "merchant_prior_failure_rate_300s_bp",
        "device_prior_failure_rate_300s_bp",
        "network_prior_failure_rate_300s_bp",
        "instrument_prior_failure_rate_300s_bp",
        "session_prior_failure_rate_300s_bp",
    }


def test_exact_window_boundaries_and_same_timestamp_order(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    engine.process(event_factory(1))
    same_timestamp = engine.process(event_factory(2))
    assert same_timestamp.features.merchant_prior_attempt_count_60s == 1
    assert same_timestamp.features.checkout_previous_attempt_age_ms == 0
    assert same_timestamp.features.checkout_history_present == 1

    at_sixty = engine.process(event_factory(3, seconds=60))
    assert at_sixty.features.merchant_prior_attempt_count_60s == 0
    assert at_sixty.features.merchant_prior_attempt_count_300s == 2

    at_three_hundred = engine.process(event_factory(4, seconds=300))
    assert at_three_hundred.features.merchant_prior_attempt_count_300s == 1
    assert at_three_hundred.features.merchant_prior_attempt_count_3600s == 3

    at_hour = engine.process(event_factory(5, seconds=3_600))
    assert at_hour.features.merchant_prior_attempt_count_3600s == 2
    assert at_hour.features.checkout_prior_attempt_count_3600s == 2


def test_strict_monotonic_order_and_typed_input(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    engine.process(event_factory(2, seconds=1))
    with pytest.raises(FeatureEngineInputError, match="monotonic"):
        engine.process(event_factory(1, seconds=1))
    with pytest.raises(FeatureEngineInputError, match="type"):
        engine.process(object())  # type: ignore[arg-type]


def test_feature_construction_failure_does_not_update_state_or_ordering_key(
    feature_config: FeatureConfig,
    event_factory: EventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FeatureEngine(feature_config)
    event = event_factory(1)
    original_compute = engine._compute_record  # noqa: SLF001

    def fail_construction(current: CheckoutAttemptEvent) -> None:
        del current
        raise ValueError("synthetic feature construction failure")

    monkeypatch.setattr(engine, "_compute_record", fail_construction)
    with pytest.raises(ValueError, match="construction failure"):
        engine.process(event)

    monkeypatch.setattr(engine, "_compute_record", original_compute)
    record = engine.process(event)
    assert record.features.merchant_prior_attempt_count_60s == 0
    assert record.features.checkout_history_present == 0


def test_missing_tokens_never_form_history_buckets(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    engine.process(event_factory(1, device_token=None, network_token=None))
    second = engine.process(event_factory(2, seconds=1, device_token=None, network_token=None))
    assert second.features.device_token_missing == 1
    assert second.features.network_token_missing == 1
    for name, value in second.features.model_dump().items():
        if name.startswith(
            ("device_prior_", "device_distinct_", "network_prior_", "network_distinct_")
        ):
            assert value == 0
    assert second.features.merchant_distinct_devices_60s == 0
    assert second.features.merchant_distinct_networks_60s == 0


def test_checkout_history_sentinel_and_millisecond_age(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    first = engine.process(event_factory(1))
    assert first.features.checkout_prior_attempt_count_3600s == 0
    assert first.features.checkout_previous_attempt_age_ms == 0
    assert first.features.checkout_history_present == 0

    second = engine.process(event_factory(2, seconds=20, milliseconds=125))
    assert second.features.checkout_prior_attempt_count_3600s == 1
    assert second.features.checkout_previous_attempt_age_ms == 20_125
    assert second.features.checkout_history_present == 1

    expired = engine.process(event_factory(3, seconds=3_620, milliseconds=125))
    assert expired.features.checkout_prior_attempt_count_3600s == 0
    assert expired.features.checkout_previous_attempt_age_ms == 0
    assert expired.features.checkout_history_present == 0


def test_failure_rates_use_floor_integer_basis_points(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    engine.process(event_factory(1, outcome=Outcome.FAILED))
    engine.process(event_factory(2, seconds=1))
    engine.process(event_factory(3, seconds=2))
    fourth = engine.process(event_factory(4, seconds=3))
    for family in ("merchant", "device", "network", "instrument", "session"):
        assert getattr(fourth.features, f"{family}_prior_failure_rate_300s_bp") == 3_333
    assert fourth.occurred_at.tzinfo is UTC
