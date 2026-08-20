from datetime import UTC, datetime, timedelta

from riskloom.features.state import FeatureState, RollingIndex

EXPECTED_INDEXES = {
    "checkout_3600s",
    "merchant_60s",
    "merchant_300s",
    "merchant_3600s",
    "device_60s",
    "device_300s",
    "device_3600s",
    "network_60s",
    "network_300s",
    "network_3600s",
    "instrument_300s",
    "instrument_3600s",
    "session_60s",
    "session_300s",
}


def test_reference_counted_distinct_values_expire_only_after_last_reference() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    index = RollingIndex("test_60s", 60, ("devices",))
    index.observe(start, "merchant", False, {"devices": "device-a"})
    index.observe(start + timedelta(seconds=30), "merchant", False, {"devices": "device-a"})

    index.evict(start + timedelta(seconds=61))
    retained = index.snapshot("merchant")
    assert retained.attempts == 1
    assert retained.distinct["devices"] == 1

    index.evict(start + timedelta(seconds=90))
    assert index.snapshot("merchant").attempts == 0
    assert index.entity_count == 0
    assert index.observation_count == 0


def test_state_evicts_expired_entities_relationships_and_checkout_history() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    state = FeatureState()
    state.observe(
        start,
        checkout="checkout-a",
        merchant="merchant-a",
        device="device-a",
        network="network-a",
        instrument="instrument-a",
        session="session-a",
        failed=True,
    )
    assert set(state._indexes) == EXPECTED_INDEXES  # noqa: SLF001
    assert all(state.index(name).observation_count == 1 for name in EXPECTED_INDEXES)
    state.evict(start + timedelta(seconds=3_600))
    for name in EXPECTED_INDEXES:
        assert state.index(name).observation_count == 0
        assert state.index(name).entity_count == 0
    assert state.index("merchant_3600s").snapshot("merchant-a").distinct == {}


def test_state_diagnostics_are_aggregate_only_and_window_bounded() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    state = FeatureState()
    for index in range(3):
        current = start + timedelta(seconds=index * 3_601)
        state.evict(current)
        state.observe(
            current,
            checkout=f"checkout-{index}",
            merchant=f"merchant-{index}",
            device=f"device-{index}",
            network=f"network-{index}",
            instrument=f"instrument-{index}",
            session=f"session-{index}",
            failed=False,
        )
    diagnostics = state.diagnostics()
    assert diagnostics["maximum_window_seconds"] == 3_600
    assert diagnostics["final_retained_observation_references"] == 14
    assert diagnostics["final_active_entity_buckets"] == 14
    indexes = diagnostics["indexes"]
    assert isinstance(indexes, dict)
    assert set(indexes) == EXPECTED_INDEXES
    assert (
        sum(item["final_observations"] for item in indexes.values())
        == diagnostics["final_retained_observation_references"]
    )
    assert (
        sum(item["final_entities"] for item in indexes.values())
        == diagnostics["final_active_entity_buckets"]
    )
    assert (
        diagnostics["peak_retained_observation_references"]
        >= diagnostics["final_retained_observation_references"]
    )
    assert diagnostics["peak_active_entity_buckets"] >= diagnostics["final_active_entity_buckets"]
    for item in indexes.values():
        assert item["peak_observations"] >= item["final_observations"]
        assert item["peak_entities"] >= item["final_entities"]
    rendered = repr(diagnostics)
    assert "merchant-" not in rendered
    assert "device-" not in rendered
