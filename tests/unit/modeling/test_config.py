import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskloom.modeling.config import (
    ModelingConfig,
    ModelingConfigurationError,
    load_modeling_config,
)


def test_default_configuration_is_locked_and_canonical(modeling_config: ModelingConfig) -> None:
    assert modeling_config.calibration_boundary_basis_points == 6_000
    assert modeling_config.false_positive_cost_units == 1
    assert modeling_config.false_negative_cost_units == 25
    assert modeling_config.source_contract.dataset_profile == "development"
    assert modeling_config.logistic_regression.random_state == 20260820
    assert modeling_config.gradient_boosting.random_state == 20260820
    assert modeling_config.platt_calibration.random_state == 20260820


@pytest.mark.parametrize(
    ("field", "value"),
    [("false_positive_cost_units", 2), ("false_negative_cost_units", 24), ("extra", 1)],
)
def test_configuration_rejects_policy_changes_and_unknown_fields(
    modeling_config: ModelingConfig, field: str, value: int
) -> None:
    raw = modeling_config.model_dump(mode="python")
    raw[field] = value
    with pytest.raises(ValidationError):
        ModelingConfig.model_validate(raw, strict=True)


@pytest.mark.parametrize("suffix", [b" ", b"\r\n"])
def test_configuration_loader_rejects_noncanonical_bytes(tmp_path: Path, suffix: bytes) -> None:
    path = tmp_path / "config.json"
    source = Path("configs/modeling/default.json").read_bytes().rstrip(b"\n")
    path.write_bytes(source + suffix)
    with pytest.raises(ModelingConfigurationError, match="not_canonical"):
        load_modeling_config(path)


def test_configuration_loader_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{\n", encoding="utf-8", newline="\n")
    with pytest.raises(ModelingConfigurationError, match="invalid"):
        load_modeling_config(path)


def test_configuration_rejects_wrong_locked_source_hash(modeling_config: ModelingConfig) -> None:
    raw = modeling_config.model_dump(mode="json")
    raw["source_contract"]["labels_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        ModelingConfig.model_validate(json.loads(json.dumps(raw)), strict=True)
