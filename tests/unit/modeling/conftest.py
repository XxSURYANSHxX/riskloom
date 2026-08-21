from pathlib import Path

import numpy as np
import pytest

from riskloom.features.schema import FEATURE_COUNT
from riskloom.modeling.config import ModelingConfig, load_modeling_config
from riskloom.modeling.data import PartitionData, TrainingData
from riskloom.modeling.training import TrainingOutputs, build_training_outputs


@pytest.fixture(scope="session")
def modeling_config() -> ModelingConfig:
    return load_modeling_config(Path("configs/modeling/default.json"))


def _partition(row_count: int, campaign_count: int = 5) -> PartitionData:
    targets = np.asarray([1 if index % 10 == 0 else 0 for index in range(row_count)], dtype=np.int8)
    features = np.zeros((row_count, FEATURE_COUNT), dtype=np.float64)
    features[:, 0] = np.arange(row_count) % 13
    features[:, 1] = targets * 25 + np.arange(row_count) % 3
    features[:, 2] = np.arange(row_count) % 7
    legitimate = (
        "normal",
        "legitimate_retry",
        "flash_sale",
        "shared_infrastructure",
        "legitimate_failure",
    )
    scenarios: list[str] = []
    campaigns: list[str | None] = []
    attack_index = 0
    for index, target in enumerate(targets):
        if target:
            scenarios.append("card_testing_campaign")
            campaigns.append(f"cmp_{attack_index % campaign_count:032x}")
            attack_index += 1
        else:
            scenarios.append(legitimate[index % len(legitimate)])
            campaigns.append(None)
    occurred_at_ms = np.arange(row_count, dtype=np.int64) * 1_000
    return PartitionData(features, targets, tuple(scenarios), tuple(campaigns), occurred_at_ms)


@pytest.fixture(scope="session")
def tiny_training_data() -> TrainingData:
    return TrainingData(
        train=_partition(120),
        calibration_fit=_partition(80),
        policy_selection=_partition(70),
        boundary_timestamp="2026-01-24T00:00:00.000Z",
    )


@pytest.fixture(scope="session")
def training_outputs(
    tiny_training_data: TrainingData, modeling_config: ModelingConfig
) -> TrainingOutputs:
    return build_training_outputs(tiny_training_data, modeling_config)
