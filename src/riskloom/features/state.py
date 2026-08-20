from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class EntitySnapshot:
    attempts: int = 0
    failures: int = 0
    distinct: Mapping[str, int] = field(default_factory=dict)
    last_occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class _Observation:
    occurred_at: datetime
    primary_token: str
    failed: bool
    relations: tuple[tuple[str, str | None], ...]
    previous_last_occurred_at: datetime | None


@dataclass(slots=True)
class _Aggregate:
    attempts: int
    failures: int
    relation_counts: dict[str, Counter[str]]
    last_occurred_at: datetime


class RollingIndex:
    """One exact entity/window index with globally ordered eviction."""

    def __init__(self, name: str, window_seconds: int, relation_names: tuple[str, ...]) -> None:
        self.name = name
        self.window_seconds = window_seconds
        self._relation_names = relation_names
        self._observations: deque[_Observation] = deque()
        self._aggregates: dict[str, _Aggregate] = {}
        self.peak_observations = 0
        self.peak_entities = 0

    def evict(self, current_time: datetime) -> tuple[_Observation, ...]:
        cutoff = current_time - timedelta(seconds=self.window_seconds)
        evicted: list[_Observation] = []
        while self._observations and self._observations[0].occurred_at <= cutoff:
            observation = self._observations.popleft()
            self._remove_aggregate_reference(observation, restore_previous_last=False)
            evicted.append(observation)
        return tuple(evicted)

    def restore_evicted(self, observations: tuple[_Observation, ...]) -> None:
        for observation in reversed(observations):
            self._observations.appendleft(observation)
        for observation in observations:
            aggregate = self._aggregates.get(observation.primary_token)
            if aggregate is None:
                aggregate = _Aggregate(
                    attempts=0,
                    failures=0,
                    relation_counts={name: Counter() for name in self._relation_names},
                    last_occurred_at=observation.occurred_at,
                )
                self._aggregates[observation.primary_token] = aggregate
            aggregate.attempts += 1
            aggregate.failures += int(observation.failed)
            aggregate.last_occurred_at = max(aggregate.last_occurred_at, observation.occurred_at)
            for relation_name, token in observation.relations:
                if token is not None:
                    aggregate.relation_counts[relation_name][token] += 1

    def snapshot(self, primary_token: str | None) -> EntitySnapshot:
        if primary_token is None:
            return EntitySnapshot()
        aggregate = self._aggregates.get(primary_token)
        if aggregate is None:
            return EntitySnapshot()
        return EntitySnapshot(
            attempts=aggregate.attempts,
            failures=aggregate.failures,
            distinct={
                relation_name: len(aggregate.relation_counts[relation_name])
                for relation_name in self._relation_names
            },
            last_occurred_at=aggregate.last_occurred_at,
        )

    def observe(
        self,
        occurred_at: datetime,
        primary_token: str | None,
        failed: bool,
        relations: Mapping[str, str | None],
    ) -> None:
        if primary_token is None:
            return
        relation_values = tuple((name, relations.get(name)) for name in self._relation_names)
        aggregate = self._aggregates.get(primary_token)
        previous_last_occurred_at = aggregate.last_occurred_at if aggregate is not None else None
        observation = _Observation(
            occurred_at,
            primary_token,
            failed,
            relation_values,
            previous_last_occurred_at,
        )
        self._observations.append(observation)
        if aggregate is None:
            aggregate = _Aggregate(
                attempts=0,
                failures=0,
                relation_counts={name: Counter() for name in self._relation_names},
                last_occurred_at=occurred_at,
            )
            self._aggregates[primary_token] = aggregate
        aggregate.attempts += 1
        aggregate.failures += int(failed)
        aggregate.last_occurred_at = occurred_at
        for relation_name, token in relation_values:
            if token is not None:
                aggregate.relation_counts[relation_name][token] += 1

    def rollback_to_observation_count(self, previous_count: int) -> None:
        if len(self._observations) == previous_count:
            return
        if len(self._observations) != previous_count + 1:
            raise RuntimeError("rolling_index_transaction_inconsistent")
        observation = self._observations.pop()
        self._remove_aggregate_reference(observation, restore_previous_last=True)

    def commit_diagnostics(self) -> None:
        self.peak_observations = max(self.peak_observations, len(self._observations))
        self.peak_entities = max(self.peak_entities, len(self._aggregates))

    def _remove_aggregate_reference(
        self, observation: _Observation, *, restore_previous_last: bool
    ) -> None:
        aggregate = self._aggregates[observation.primary_token]
        if aggregate.attempts <= 0 or (observation.failed and aggregate.failures <= 0):
            raise RuntimeError("rolling_index_counter_inconsistent")
        aggregate.attempts -= 1
        if observation.failed:
            aggregate.failures -= 1
        for relation_name, token in observation.relations:
            if token is None:
                continue
            counts = aggregate.relation_counts[relation_name]
            if counts[token] <= 0:
                raise RuntimeError("rolling_index_relation_counter_inconsistent")
            counts[token] -= 1
            if counts[token] == 0:
                del counts[token]
        if aggregate.attempts == 0:
            if aggregate.failures != 0 or any(aggregate.relation_counts.values()):
                raise RuntimeError("rolling_index_terminal_counter_inconsistent")
            del self._aggregates[observation.primary_token]
        elif restore_previous_last:
            if observation.previous_last_occurred_at is None:
                raise RuntimeError("rolling_index_previous_timestamp_missing")
            aggregate.last_occurred_at = observation.previous_last_occurred_at

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def entity_count(self) -> int:
        return len(self._aggregates)

    def diagnostics(self) -> dict[str, int]:
        return {
            "final_entities": self.entity_count,
            "final_observations": self.observation_count,
            "peak_entities": self.peak_entities,
            "peak_observations": self.peak_observations,
        }


class FeatureState:
    """All fixed Day 3 rolling indexes and aggregate-only diagnostics."""

    def __init__(self) -> None:
        specifications = (
            ("checkout_3600s", 3_600, ()),
            ("merchant_60s", 60, ("instruments", "devices", "networks")),
            ("merchant_300s", 300, ("instruments", "devices", "networks")),
            ("merchant_3600s", 3_600, ("instruments", "devices", "networks")),
            ("device_60s", 60, ("instruments", "sessions")),
            ("device_300s", 300, ("instruments", "sessions")),
            ("device_3600s", 3_600, ("instruments", "sessions")),
            ("network_60s", 60, ("instruments", "devices", "sessions")),
            ("network_300s", 300, ("instruments", "devices", "sessions")),
            ("network_3600s", 3_600, ("instruments", "devices", "sessions")),
            ("instrument_300s", 300, ("devices", "networks", "merchants")),
            ("instrument_3600s", 3_600, ("devices", "networks", "merchants")),
            ("session_60s", 60, ("instruments",)),
            ("session_300s", 300, ("instruments",)),
        )
        self._indexes = {
            name: RollingIndex(name, seconds, relation_names)
            for name, seconds, relation_names in specifications
        }
        self._peak_observation_references = 0
        self._peak_entity_buckets = 0

    def index(self, name: str) -> RollingIndex:
        return self._indexes[name]

    def evict(self, current_time: datetime) -> dict[str, tuple[_Observation, ...]]:
        return {name: index.evict(current_time) for name, index in self._indexes.items()}

    def restore_eviction(self, checkpoint: Mapping[str, tuple[_Observation, ...]]) -> None:
        for name, observations in checkpoint.items():
            self._indexes[name].restore_evicted(observations)

    def observe(
        self,
        occurred_at: datetime,
        *,
        checkout: str,
        merchant: str,
        device: str | None,
        network: str | None,
        instrument: str,
        session: str,
        failed: bool,
    ) -> None:
        operations: list[tuple[RollingIndex, str | None, Mapping[str, str | None]]] = []
        for window in (60, 300, 3_600):
            operations.extend(
                (
                    (
                        self.index(f"merchant_{window}s"),
                        merchant,
                        {"instruments": instrument, "devices": device, "networks": network},
                    ),
                    (
                        self.index(f"device_{window}s"),
                        device,
                        {"instruments": instrument, "sessions": session},
                    ),
                    (
                        self.index(f"network_{window}s"),
                        network,
                        {"instruments": instrument, "devices": device, "sessions": session},
                    ),
                )
            )
        for window in (300, 3_600):
            operations.append(
                (
                    self.index(f"instrument_{window}s"),
                    instrument,
                    {"devices": device, "networks": network, "merchants": merchant},
                )
            )
        for window in (60, 300):
            operations.append(
                (
                    self.index(f"session_{window}s"),
                    session,
                    {"instruments": instrument},
                )
            )
        operations.append((self.index("checkout_3600s"), checkout, {}))

        attempted: list[tuple[RollingIndex, int]] = []
        previous_index_peaks = tuple(
            (index, index.peak_observations, index.peak_entities) for index, _, _ in operations
        )
        previous_peak_observations = self._peak_observation_references
        previous_peak_entities = self._peak_entity_buckets
        try:
            for index, primary_token, relations in operations:
                previous_count = index.observation_count
                attempted.append((index, previous_count))
                index.observe(occurred_at, primary_token, failed, relations)
            for index, _, _ in operations:
                index.commit_diagnostics()
            observation_references = sum(
                index.observation_count for index in self._indexes.values()
            )
            entity_buckets = sum(index.entity_count for index in self._indexes.values())
            self._peak_observation_references = max(
                self._peak_observation_references, observation_references
            )
            self._peak_entity_buckets = max(self._peak_entity_buckets, entity_buckets)
        except Exception:
            for index, previous_count in reversed(attempted):
                index.rollback_to_observation_count(previous_count)
            for index, peak_observations, peak_entities in previous_index_peaks:
                index.peak_observations = peak_observations
                index.peak_entities = peak_entities
            self._peak_observation_references = previous_peak_observations
            self._peak_entity_buckets = previous_peak_entities
            raise

    def diagnostics(self) -> dict[str, object]:
        final_observations = sum(index.observation_count for index in self._indexes.values())
        final_entities = sum(index.entity_count for index in self._indexes.values())
        return {
            "final_active_entity_buckets": final_entities,
            "final_retained_observation_references": final_observations,
            "indexes": {name: self._indexes[name].diagnostics() for name in sorted(self._indexes)},
            "maximum_window_seconds": 3_600,
            "peak_active_entity_buckets": self._peak_entity_buckets,
            "peak_retained_observation_references": self._peak_observation_references,
        }
