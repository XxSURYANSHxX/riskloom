from collections.abc import Callable, Mapping

from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine
from riskloom.features.schema import FeatureRecord
from riskloom.simulation.event_schema import CheckoutAttemptEvent, Outcome

EventFactory = Callable[..., CheckoutAttemptEvent]


def _assert_fields(record: FeatureRecord, expected: Mapping[str, int]) -> None:
    actual = record.features.model_dump()
    for name, value in expected.items():
        assert actual[name] == value, name


def test_hand_computed_causal_oracle_covers_all_families_and_boundaries(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    merchant = f"mrc_{'a' * 32}"
    device_1 = f"dev_{'b' * 32}"
    device_2 = f"dev_{'c' * 32}"
    network = f"net_{'d' * 32}"
    instrument_1 = f"pmt_{'a' * 32}"
    instrument_2 = f"pmt_{'b' * 32}"
    session_1 = f"ses_{'c' * 32}"
    session_2 = f"ses_{'d' * 32}"
    checkout_1 = f"chk_{'a' * 32}"
    checkout_2 = f"chk_{'b' * 32}"
    checkout_3 = f"chk_{'c' * 32}"
    checkout_4 = f"chk_{'d' * 32}"
    checkout_5 = f"chk_{'e' * 32}"

    events = (
        event_factory(
            1,
            merchant_id=merchant,
            checkout_id=checkout_1,
            device_token=device_1,
            network_token=network,
            payment_instrument_token=instrument_1,
            session_token=session_1,
        ),
        event_factory(
            2,
            seconds=30,
            outcome=Outcome.FAILED,
            merchant_id=merchant,
            checkout_id=checkout_1,
            device_token=device_1,
            network_token=network,
            payment_instrument_token=instrument_1,
            session_token=session_1,
        ),
        event_factory(
            3,
            seconds=60,
            merchant_id=merchant,
            checkout_id=checkout_2,
            device_token=device_1,
            network_token=network,
            payment_instrument_token=instrument_1,
            session_token=session_1,
        ),
        event_factory(
            4,
            seconds=60,
            merchant_id=merchant,
            checkout_id=checkout_3,
            device_token=device_2,
            network_token=network,
            payment_instrument_token=instrument_2,
            session_token=session_2,
        ),
        event_factory(
            5,
            seconds=90,
            outcome=Outcome.FAILED,
            merchant_id=merchant,
            checkout_id=checkout_3,
            device_token=None,
            network_token=None,
            payment_instrument_token=instrument_2,
            session_token=session_2,
        ),
        event_factory(
            6,
            seconds=300,
            merchant_id=merchant,
            checkout_id=checkout_4,
            device_token=device_1,
            network_token=network,
            payment_instrument_token=instrument_1,
            session_token=session_1,
        ),
        event_factory(
            7,
            seconds=3_600,
            merchant_id=merchant,
            checkout_id=checkout_5,
            device_token=device_1,
            network_token=network,
            payment_instrument_token=instrument_1,
            session_token=session_1,
        ),
    )
    rows = tuple(engine.process(event) for event in events)

    _assert_fields(
        rows[0],
        {
            "amount_subunits": 10_001,
            "utc_hour": 0,
            "utc_day_of_week": 3,
            "channel_web": 1,
            "channel_mobile_web": 0,
            "channel_mobile_app": 0,
            "customer_token_missing": 0,
            "device_token_missing": 0,
            "network_token_missing": 0,
            "merchant_prior_attempt_count_60s": 0,
        },
    )
    _assert_fields(
        rows[1],
        {
            "checkout_prior_attempt_count_3600s": 1,
            "checkout_history_present": 1,
            "checkout_previous_attempt_age_ms": 30_000,
            "merchant_prior_attempt_count_60s": 1,
            "device_prior_attempt_count_60s": 1,
            "network_prior_attempt_count_60s": 1,
            "instrument_prior_attempt_count_300s": 1,
            "session_prior_attempt_count_60s": 1,
        },
    )
    _assert_fields(
        rows[2],
        {
            "checkout_prior_attempt_count_3600s": 0,
            "checkout_history_present": 0,
            "checkout_previous_attempt_age_ms": 0,
            "merchant_prior_attempt_count_60s": 1,
            "merchant_prior_failure_count_60s": 1,
            "merchant_prior_attempt_count_300s": 2,
            "merchant_prior_failure_count_300s": 1,
            "merchant_distinct_instruments_300s": 1,
            "merchant_distinct_devices_300s": 1,
            "merchant_distinct_networks_300s": 1,
            "device_prior_attempt_count_60s": 1,
            "device_prior_failure_count_60s": 1,
            "device_distinct_instruments_300s": 1,
            "device_distinct_sessions_300s": 1,
            "network_prior_attempt_count_60s": 1,
            "network_prior_failure_count_60s": 1,
            "network_distinct_instruments_300s": 1,
            "network_distinct_devices_300s": 1,
            "network_distinct_sessions_300s": 1,
            "instrument_prior_attempt_count_300s": 2,
            "instrument_prior_failure_count_300s": 1,
            "instrument_distinct_devices_300s": 1,
            "instrument_distinct_networks_300s": 1,
            "instrument_distinct_merchants_300s": 1,
            "session_prior_attempt_count_60s": 1,
            "session_prior_failure_count_60s": 1,
            "session_distinct_instruments_300s": 1,
            "merchant_prior_failure_rate_300s_bp": 5_000,
            "device_prior_failure_rate_300s_bp": 5_000,
            "network_prior_failure_rate_300s_bp": 5_000,
            "instrument_prior_failure_rate_300s_bp": 5_000,
            "session_prior_failure_rate_300s_bp": 5_000,
        },
    )
    _assert_fields(
        rows[3],
        {
            "merchant_prior_attempt_count_60s": 2,
            "merchant_prior_failure_count_60s": 1,
            "network_prior_attempt_count_60s": 2,
            "device_prior_attempt_count_60s": 0,
            "instrument_prior_attempt_count_300s": 0,
            "session_prior_attempt_count_60s": 0,
            "merchant_prior_failure_rate_300s_bp": 3_333,
        },
    )
    _assert_fields(
        rows[4],
        {
            "device_token_missing": 1,
            "network_token_missing": 1,
            "checkout_prior_attempt_count_3600s": 1,
            "checkout_history_present": 1,
            "checkout_previous_attempt_age_ms": 30_000,
            "merchant_prior_attempt_count_60s": 2,
            "merchant_prior_failure_count_60s": 0,
            "merchant_distinct_instruments_60s": 2,
            "merchant_distinct_devices_60s": 2,
            "merchant_distinct_networks_60s": 1,
            "device_prior_attempt_count_300s": 0,
            "network_prior_attempt_count_300s": 0,
            "instrument_prior_attempt_count_300s": 1,
            "instrument_distinct_devices_300s": 1,
            "instrument_distinct_networks_300s": 1,
            "instrument_distinct_merchants_300s": 1,
            "session_prior_attempt_count_300s": 1,
            "session_distinct_instruments_300s": 1,
            "merchant_prior_failure_rate_300s_bp": 2_500,
            "device_prior_failure_rate_300s_bp": 0,
            "network_prior_failure_rate_300s_bp": 0,
            "instrument_prior_failure_rate_300s_bp": 0,
            "session_prior_failure_rate_300s_bp": 0,
        },
    )
    _assert_fields(
        rows[5],
        {
            "merchant_prior_attempt_count_60s": 0,
            "merchant_prior_attempt_count_300s": 4,
            "merchant_prior_failure_count_300s": 2,
            "merchant_distinct_instruments_300s": 2,
            "merchant_distinct_devices_300s": 2,
            "merchant_distinct_networks_300s": 1,
            "merchant_prior_attempt_count_3600s": 5,
            "device_prior_attempt_count_300s": 2,
            "device_prior_failure_count_300s": 1,
            "device_prior_attempt_count_3600s": 3,
            "network_prior_attempt_count_300s": 3,
            "network_prior_failure_count_300s": 1,
            "network_distinct_instruments_300s": 2,
            "network_distinct_devices_300s": 2,
            "network_distinct_sessions_300s": 2,
            "instrument_prior_attempt_count_300s": 2,
            "instrument_prior_attempt_count_3600s": 3,
            "session_prior_attempt_count_300s": 2,
            "merchant_prior_failure_rate_300s_bp": 5_000,
            "device_prior_failure_rate_300s_bp": 5_000,
            "network_prior_failure_rate_300s_bp": 3_333,
            "instrument_prior_failure_rate_300s_bp": 5_000,
            "session_prior_failure_rate_300s_bp": 5_000,
        },
    )
    _assert_fields(
        rows[6],
        {
            "merchant_prior_attempt_count_300s": 0,
            "merchant_prior_attempt_count_3600s": 5,
            "merchant_prior_failure_count_3600s": 2,
            "merchant_distinct_instruments_3600s": 2,
            "merchant_distinct_devices_3600s": 2,
            "merchant_distinct_networks_3600s": 1,
            "device_prior_attempt_count_3600s": 3,
            "device_prior_failure_count_3600s": 1,
            "network_prior_attempt_count_3600s": 4,
            "network_prior_failure_count_3600s": 1,
            "instrument_prior_attempt_count_3600s": 3,
            "instrument_prior_failure_count_3600s": 1,
            "session_prior_attempt_count_300s": 0,
            "merchant_prior_failure_rate_300s_bp": 0,
            "device_prior_failure_rate_300s_bp": 0,
            "network_prior_failure_rate_300s_bp": 0,
            "instrument_prior_failure_rate_300s_bp": 0,
            "session_prior_failure_rate_300s_bp": 0,
        },
    )
