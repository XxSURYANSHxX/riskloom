from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import pytest

from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine, FeatureEngineInputError
from riskloom.features.schema import FeatureVector
from riskloom.features.state import RollingIndex
from riskloom.simulation.event_schema import CheckoutAttemptEvent

EventFactory = Callable[..., CheckoutAttemptEvent]


def _index_signature(index: RollingIndex) -> tuple[object, ...]:
    aggregates = tuple(
        (
            token,
            aggregate.attempts,
            aggregate.failures,
            aggregate.last_occurred_at,
            tuple(
                (name, tuple(sorted(counts.items())))
                for name, counts in sorted(aggregate.relation_counts.items())
            ),
        )
        for token, aggregate in sorted(index._aggregates.items())  # noqa: SLF001
    )
    return (
        tuple(index._observations),  # noqa: SLF001
        aggregates,
        index.peak_entities,
        index.peak_observations,
    )


def _engine_signature(engine: FeatureEngine) -> tuple[object, ...]:
    indexes = engine._state._indexes  # noqa: SLF001
    return (
        engine._last_key,  # noqa: SLF001
        tuple((name, _index_signature(indexes[name])) for name in sorted(indexes)),
        engine.diagnostics(),
    )


def test_invalid_type_and_order_leave_every_state_component_unchanged(
    feature_config: FeatureConfig, event_factory: EventFactory
) -> None:
    engine = FeatureEngine(feature_config)
    engine.process(event_factory(2, seconds=2))
    before = _engine_signature(engine)

    with pytest.raises(FeatureEngineInputError, match="type"):
        engine.process(object())  # type: ignore[arg-type]
    assert _engine_signature(engine) == before

    with pytest.raises(FeatureEngineInputError, match="monotonic"):
        engine.process(event_factory(1, seconds=1))
    assert _engine_signature(engine) == before


@pytest.mark.parametrize("failure_kind", ["construction", "validation"])
def test_preupdate_failure_restores_eviction_and_ordering_key(
    failure_kind: str,
    feature_config: FeatureConfig,
    event_factory: EventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FeatureEngine(feature_config)
    initial = event_factory(1)
    engine.process(initial)
    before = _engine_signature(engine)
    expired_candidate = event_factory(3, seconds=3_600)

    if failure_kind == "construction":

        def fail_construction(event: CheckoutAttemptEvent) -> None:
            del event
            raise ValueError("synthetic construction failure")

        monkeypatch.setattr(engine, "_compute_record", fail_construction)
    else:

        def fail_validation(cls: type[FeatureVector], value: Mapping[str, Any]) -> None:
            del cls, value
            raise ValueError("synthetic validation failure")

        monkeypatch.setattr(FeatureVector, "model_validate", classmethod(fail_validation))

    with pytest.raises(ValueError, match=failure_kind):
        engine.process(expired_candidate)
    assert _engine_signature(engine) == before

    monkeypatch.undo()
    retry = engine.process(event_factory(2, seconds=1))
    assert retry.features.merchant_prior_attempt_count_60s == 1
    assert retry.features.checkout_prior_attempt_count_3600s == 1


def test_late_index_failure_rolls_back_all_current_event_updates(
    feature_config: FeatureConfig,
    event_factory: EventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FeatureEngine(feature_config)
    reference = FeatureEngine(feature_config)
    initial = event_factory(1)
    candidate = event_factory(2, seconds=1)
    engine.process(initial)
    reference.process(initial)
    before = _engine_signature(engine)

    target = engine._state.index("checkout_3600s")  # noqa: SLF001
    original_observe = target.observe

    def fail_after_observe(
        occurred_at: datetime,
        primary_token: str | None,
        failed: bool,
        relations: Mapping[str, str | None],
    ) -> None:
        original_observe(occurred_at, primary_token, failed, relations)
        raise RuntimeError("synthetic late-index failure")

    monkeypatch.setattr(target, "observe", fail_after_observe)
    with pytest.raises(RuntimeError, match="late-index"):
        engine.process(candidate)
    assert _engine_signature(engine) == before

    monkeypatch.undo()
    assert engine.process(candidate) == reference.process(candidate)
    assert engine.diagnostics() == reference.diagnostics()


def test_diagnostic_commit_failure_rolls_back_state_and_peak_bookkeeping(
    feature_config: FeatureConfig,
    event_factory: EventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FeatureEngine(feature_config)
    initial = event_factory(1)
    candidate = event_factory(2, seconds=1)
    engine.process(initial)
    before = _engine_signature(engine)
    target = engine._state.index("network_300s")  # noqa: SLF001
    original_commit = target.commit_diagnostics

    def fail_after_diagnostic_commit() -> None:
        original_commit()
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(target, "commit_diagnostics", fail_after_diagnostic_commit)
    with pytest.raises(RuntimeError, match="diagnostic"):
        engine.process(candidate)
    assert _engine_signature(engine) == before

    monkeypatch.undo()
    assert engine.process(candidate).features.merchant_prior_attempt_count_60s == 1
