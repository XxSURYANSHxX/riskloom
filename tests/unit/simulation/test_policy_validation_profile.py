"""The locked contract for the Day 5 counterfactual batch profile."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from riskloom.simulation.config import (
    POLICY_VALIDATION_TOTAL_EVENTS,
    GeneratorConfig,
    load_generator_config,
    validate_profile_contract,
)

CONFIG_PATH = Path("configs/simulation/policy-validation.json")


def _raw() -> dict[str, Any]:
    import json  # noqa: PLC0415

    return dict(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def test_shipped_policy_validation_profile_satisfies_its_locked_contract() -> None:
    config = load_generator_config(CONFIG_PATH)
    assert config.dataset_profile == "policy-validation"
    assert config.config_schema_version == "1.1.0"
    assert config.start_at == datetime(2026, 3, 1, tzinfo=UTC)
    assert config.total_events == POLICY_VALIDATION_TOTAL_EVENTS == 9_000
    assert [split.event_count for split in config.splits] == [5_000, 2_000, 2_000]
    assert [split.duration_days for split in config.splits] == [5, 3, 3]
    assert [split.campaign_count for split in config.splits] == [5, 4, 2]
    assert sum(config.scenario_counts(split)["attack"] for split in config.splits) == 180


def test_policy_validation_profile_is_not_confusable_with_development() -> None:
    policy_validation = load_generator_config(CONFIG_PATH)
    development = load_generator_config(Path("configs/simulation/development.json"))
    assert policy_validation.dataset_profile != development.dataset_profile
    assert policy_validation.total_events != development.total_events
    assert policy_validation.start_at != development.start_at


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"start_at": "2026-01-01T00:00:00Z"}, "policy_validation_contract_start_at"),
        ({"__train_event_count": 6_000}, "policy_validation_contract_train_event_count"),
        ({"__train_duration": 6}, "policy_validation_contract_train_duration"),
        ({"__calibration_campaigns": 5}, "policy_validation_contract_calibration_campaign_count"),
        ({"__test_event_count": 3_000}, "policy_validation_contract_test_event_count"),
        ({"__boundary": 6_000}, "policy_validation_contract_boundary"),
        ({"__gap": 600}, "policy_validation_contract_campaign_gap"),
    ],
)
def test_locked_policy_validation_contract_rejects_reshaping(
    mutation: dict[str, Any], message: str
) -> None:
    raw = _raw()
    for key, value in mutation.items():
        if key == "__train_event_count":
            raw["splits"][0]["event_count"] = value
        elif key == "__train_duration":
            raw["splits"][0]["duration_days"] = value
        elif key == "__calibration_campaigns":
            raw["splits"][1]["campaign_count"] = value
        elif key == "__test_event_count":
            raw["splits"][2]["event_count"] = value
        elif key == "__boundary":
            raw["splits"][1]["campaign_placement"]["protected_boundary_basis_points"] = value
        elif key == "__gap":
            raw["splits"][1]["campaign_placement"]["minimum_gap_seconds"] = value
        else:
            raw[key] = value
    with pytest.raises(ValueError, match=message):
        GeneratorConfig.model_validate(raw)


def test_policy_validation_profile_requires_schema_1_1_0() -> None:
    raw = _raw()
    raw["config_schema_version"] = "1.0.0"
    raw["splits"][1].pop("campaign_placement")
    with pytest.raises(ValueError, match="unsupported_version_profile_combination"):
        GeneratorConfig.model_validate(raw)


def test_unknown_dataset_profile_is_still_refused() -> None:
    raw = _raw()
    raw["dataset_profile"] = "production"
    with pytest.raises(ValueError):
        GeneratorConfig.model_validate(raw)


def test_development_and_smoke_contracts_are_unaffected() -> None:
    for path in (
        Path("configs/simulation/development.json"),
        Path("configs/simulation/smoke.json"),
    ):
        config = load_generator_config(path)
        validate_profile_contract(config)
        assert config.dataset_profile in {"development", "smoke"}
