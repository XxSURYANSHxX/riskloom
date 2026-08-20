from pathlib import Path

import pytest

from riskloom.simulation.artifacts import GenerationResult, generate_dataset
from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.generation import GeneratedRecord, generate_records


@pytest.fixture
def tiny_config() -> GeneratorConfig:
    return GeneratorConfig.model_validate(
        {
            "config_schema_version": "1.0.0",
            "dataset_profile": "smoke",
            "start_at": "2026-01-01T00:00:00Z",
            "currency": "INR",
            "merchant_count": 3,
            "entity_pools": {
                "customers": 60,
                "devices": 45,
                "networks": 15,
                "instruments": 80,
            },
            "splits": [
                {
                    "name": "train",
                    "duration_days": 1,
                    "event_count": 100,
                    "campaign_count": 1,
                    "campaign_profile": "baseline_reuse",
                },
                {
                    "name": "calibration",
                    "duration_days": 1,
                    "event_count": 100,
                    "campaign_count": 1,
                    "campaign_profile": "baseline_reuse",
                },
                {
                    "name": "test",
                    "duration_days": 1,
                    "event_count": 100,
                    "campaign_count": 1,
                    "campaign_profile": "entity_reuse_shift",
                },
            ],
        }
    )


@pytest.fixture
def tiny_records(tiny_config: GeneratorConfig) -> list[GeneratedRecord]:
    return generate_records(tiny_config, 1_234)


@pytest.fixture
def tiny_output(tmp_path: Path, tiny_config: GeneratorConfig) -> tuple[Path, GenerationResult]:
    output = tmp_path / "simulation"
    return output, generate_dataset(tiny_config, 1_234, output)
