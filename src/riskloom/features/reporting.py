from collections import Counter
from typing import Any

from riskloom.features.config import FEATURE_SCHEMA_VERSION
from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES, FeatureRecord


class FeatureStatistics:
    """Streaming exact integer distributions without retaining feature rows."""

    def __init__(self) -> None:
        self._frequencies: dict[str, Counter[int]] = {name: Counter() for name in FEATURE_NAMES}
        self.row_count = 0

    def add(self, record: FeatureRecord) -> None:
        for name in FEATURE_NAMES:
            self._frequencies[name][getattr(record.features, name)] += 1
        self.row_count += 1

    @staticmethod
    def _nearest_rank(frequencies: Counter[int], percentile: int, count: int) -> int:
        target = max(1, (percentile * count + 99) // 100)
        cumulative = 0
        for value in sorted(frequencies):
            cumulative += frequencies[value]
            if cumulative >= target:
                return value
        raise RuntimeError("feature_percentile_unreachable")

    def distributions(self) -> dict[str, dict[str, int]]:
        if self.row_count == 0:
            raise ValueError("feature_statistics_empty")
        result: dict[str, dict[str, int]] = {}
        for name in FEATURE_NAMES:
            frequencies = self._frequencies[name]
            values = frequencies.keys()
            result[name] = {
                "max": max(values),
                "min": min(values),
                "p50": self._nearest_rank(frequencies, 50, self.row_count),
                "p95": self._nearest_rank(frequencies, 95, self.row_count),
                "p99": self._nearest_rank(frequencies, 99, self.row_count),
                "zero_count": frequencies[0],
            }
        return result


def build_feature_report(
    dataset_id: str,
    statistics: FeatureStatistics,
    state_diagnostics: dict[str, object],
) -> dict[str, Any]:
    return {
        "artifact_type": "temporal_coordination_feature_report",
        "feature_count": FEATURE_COUNT,
        "feature_dataset_id": dataset_id,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": statistics.distributions(),
        "product": "RiskLoom",
        "row_count": statistics.row_count,
        "state_diagnostics": state_diagnostics,
    }
