from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine
from riskloom.features.reporting import FeatureStatistics, build_feature_report
from riskloom.features.schema import FEATURE_COUNT
from riskloom.simulation.event_schema import CheckoutAttemptEvent


def test_report_uses_exact_nearest_rank_integer_statistics(
    feature_config: FeatureConfig,
    tiny_events: list[CheckoutAttemptEvent],
) -> None:
    engine = FeatureEngine(feature_config)
    statistics = FeatureStatistics()
    for event in tiny_events:
        statistics.add(engine.process(event))
    report = build_feature_report("a" * 64, statistics, engine.diagnostics())

    assert report["row_count"] == 12
    assert report["feature_count"] == FEATURE_COUNT
    amounts = report["features"]["amount_subunits"]
    assert amounts == {
        "max": 10_012,
        "min": 10_001,
        "p50": 10_006,
        "p95": 10_012,
        "p99": 10_012,
        "zero_count": 0,
    }
    assert report["features"]["device_token_missing"]["zero_count"] == 12
    assert all(type(value) is int for value in amounts.values())
    assert "event_id" not in repr(report["state_diagnostics"])
    assert all(
        type(value) is int and type(frequency) is int
        for frequencies in statistics._frequencies.values()  # noqa: SLF001
        for value, frequency in frequencies.items()
    )
    assert not any(
        event.event_id in repr(statistics._frequencies)  # noqa: SLF001
        for event in tiny_events
    )


def test_empty_statistics_are_rejected() -> None:
    statistics = FeatureStatistics()
    try:
        statistics.distributions()
    except ValueError as error:
        assert str(error) == "feature_statistics_empty"
    else:
        raise AssertionError("empty feature statistics must fail")
