"""End-to-end behaviour of the band fitting and counterfactual validation workflows."""

from pathlib import Path

import numpy as np
import pytest

import riskloom.modeling.policy_ops as policy_ops
from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES
from riskloom.modeling.config import ModelingConfig
from riskloom.modeling.data import PartitionData, PolicyValidationData, TrainingData, ValidationData
from riskloom.modeling.model import LockedModel, LogisticPortableModel, PlattModel
from riskloom.modeling.policy_ops import (
    PolicyOperationError,
    fit_policy_band,
    load_locked_band,
    validate_policy,
)
from riskloom.policy.artifacts import publish_band
from riskloom.policy.canonical import PolicyArtifactError, canonical_json_bytes, file_metadata
from riskloom.policy.config import PolicyConfig, load_policy_config

INCUMBENT_THRESHOLD = 0.5


@pytest.fixture(scope="session")
def policy_config() -> PolicyConfig:
    return load_policy_config(Path("configs/policy/default.json"))


def _locked_model() -> LockedModel:
    return LockedModel(
        model_id="a" * 64,
        feature_order=list(FEATURE_NAMES),
        class_order=[0, 1],
        decision_threshold=INCUMBENT_THRESHOLD,
        candidate=LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=[0.0] * FEATURE_COUNT,
            intercept=0.0,
            scaler_mean=[0.0] * FEATURE_COUNT,
            scaler_scale=[1.0] * FEATURE_COUNT,
        ),
        calibration=PlattModel(coefficient=1.0, intercept=0.0, probability_clip_epsilon=1e-15),
    )


def _partition(rows: int) -> PartitionData:
    targets = np.asarray([1 if index % 10 == 0 else 0 for index in range(rows)], dtype=np.int8)
    return PartitionData(
        features=np.zeros((rows, FEATURE_COUNT), dtype=np.float64),
        targets=targets,
        scenarios=tuple("normal" for _ in range(rows)),
        campaign_ids=tuple(None for _ in range(rows)),
        occurred_at_ms=np.arange(rows, dtype=np.int64),
    )


def _probabilities(targets: np.ndarray) -> np.ndarray:
    """Well-separated scores so the sweep has a clear optimum."""

    return np.where(np.asarray(targets) == 1, 0.9, 0.05)


def _patch_common(
    monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig, policy_config: PolicyConfig
) -> LockedModel:
    model = _locked_model()
    monkeypatch.setattr(policy_ops, "load_modeling_config", lambda _: modeling_config)
    monkeypatch.setattr(policy_ops, "load_policy_config", lambda _: policy_config)
    monkeypatch.setattr(policy_ops, "load_locked_model", lambda *_: (model, {}, {}))
    monkeypatch.setattr(
        policy_ops,
        "portable_probabilities",
        lambda _model, features: np.full(features.shape[0], 0.05),
    )
    return model


def test_fit_policy_band_publishes_a_band_from_policy_selection_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    policy_config: PolicyConfig,
) -> None:
    model = _patch_common(monkeypatch, modeling_config, policy_config)
    policy_selection = _partition(200)
    seen: dict[str, object] = {}

    def _probabilities_for(_model: LockedModel, features: np.ndarray) -> np.ndarray:
        seen["rows"] = features.shape[0]
        return _probabilities(policy_selection.targets)

    monkeypatch.setattr(policy_ops, "portable_probabilities", _probabilities_for)
    training = TrainingData(
        train=_partition(50),
        calibration_fit=_partition(30),
        policy_selection=policy_selection,
        boundary_timestamp="2026-01-24T00:00:00.000Z",
    )
    monkeypatch.setattr(policy_ops, "load_training_data", lambda *_a, **_k: training)

    result = fit_policy_band(
        Path("sim"),
        Path("feat"),
        Path("config.json"),
        Path("model"),
        Path("policy.json"),
        tmp_path / "band",
    )
    # Only the policy_selection partition is ever scored.
    assert seen["rows"] == policy_selection.row_count
    band, payload = load_locked_band(tmp_path / "band", policy_config)
    assert payload["band_id"] == result.artifact_id
    assert payload["fit_partition"] == "policy_selection"
    assert payload["model_id"] == model.model_id
    assert payload["incumbent_decision_threshold"] == INCUMBENT_THRESHOLD
    assert payload["fit_partition_outcome"]["row_count"] == policy_selection.row_count
    assert payload["tie_break_order"][0] == "minimum_cost"
    assert band.lower_threshold <= band.upper_threshold
    # The sweep may never do worse than the incumbent on the partition it fits.
    assert (
        payload["fit_partition_outcome"]["cost_units"]
        <= payload["fit_partition_incumbent_outcome"]["cost_units"]
    )


def test_fit_policy_band_refuses_a_validation_mode_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    policy_config: PolicyConfig,
) -> None:
    _patch_common(monkeypatch, modeling_config, policy_config)
    training = TrainingData(
        train=_partition(20),
        calibration_fit=_partition(20),
        policy_selection=_partition(20),
        boundary_timestamp="2026-01-24T00:00:00.000Z",
    )
    monkeypatch.setattr(
        policy_ops,
        "load_training_data",
        lambda *_a, **_k: ValidationData(
            training=training, held_out_feature_sample=np.zeros((3, FEATURE_COUNT))
        ),
    )
    with pytest.raises(PolicyOperationError, match="training_data_mode_invalid"):
        fit_policy_band(
            Path("sim"),
            Path("feat"),
            Path("config.json"),
            Path("model"),
            Path("policy.json"),
            tmp_path / "band",
        )


def _published_band(tmp_path: Path, policy_config: PolicyConfig, model_id: str = "a" * 64) -> Path:
    directory = tmp_path / "band"
    publish_band(
        directory,
        {
            "artifact_type": "cost_aware_policy_band",
            "band": {"lower_threshold": 0.2, "upper_threshold": 0.8},
            "model_id": model_id,
            "product": "RiskLoom",
        },
        policy_config,
        {"model_id": model_id},
    )
    return directory


def test_load_locked_band_round_trips(tmp_path: Path, policy_config: PolicyConfig) -> None:
    band, payload = load_locked_band(_published_band(tmp_path, policy_config), policy_config)
    assert (band.lower_threshold, band.upper_threshold) == (0.2, 0.8)
    assert payload["model_id"] == "a" * 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "directory_unreadable"),
        ("artifact_set", "artifact_set_invalid"),
        ("marker", "manifest_marker_invalid"),
        ("configuration", "configuration_mismatch"),
        ("configuration_hash", "configuration_hash_mismatch"),
        ("artifacts_shape", "manifest_artifacts_invalid"),
        ("artifact_hash", "artifact_hash_mismatch"),
        ("band_shape", "band_schema_invalid"),
    ],
)
def test_load_locked_band_fails_closed_on_tampering(
    tmp_path: Path, policy_config: PolicyConfig, mutation: str, message: str
) -> None:
    import json  # noqa: PLC0415

    if mutation == "missing":
        with pytest.raises(PolicyArtifactError, match=message):
            load_locked_band(tmp_path / "absent", policy_config)
        return

    directory = _published_band(tmp_path, policy_config)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    band = json.loads((directory / "band.json").read_bytes())

    if mutation == "artifact_set":
        (directory / "unexpected.json").write_bytes(b"{}\n")
    elif mutation == "marker":
        manifest["product"] = "NotRiskLoom"
    elif mutation == "configuration":
        manifest["effective_policy_configuration"]["review_cost_units"] = 99
    elif mutation == "configuration_hash":
        manifest["effective_policy_configuration_sha256"] = "0" * 64
    elif mutation == "artifacts_shape":
        manifest["artifacts"] = []
    elif mutation == "artifact_hash":
        manifest["artifacts"]["band.json"]["sha256"] = "0" * 64
    else:
        band["band"] = {"lower_threshold": 0.9, "upper_threshold": 0.1}
        (directory / "band.json").write_bytes(canonical_json_bytes(band))
        manifest["artifacts"]["band.json"] = {
            "byte_size": len(canonical_json_bytes(band)),
            "sha256": file_metadata(directory / "band.json")["sha256"],
        }

    if mutation != "artifact_set":
        (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(PolicyArtifactError, match=message):
        load_locked_band(directory, policy_config)


def _validation_batch(rows: int, attacks: int) -> PolicyValidationData:
    targets = np.asarray([1] * attacks + [0] * (rows - attacks), dtype=np.int8)
    return PolicyValidationData(
        features=np.zeros((rows, FEATURE_COUNT), dtype=np.float64),
        targets=targets,
        row_count=rows,
        simulation_dataset_id="1" * 64,
        feature_dataset_id="2" * 64,
        events_sha256="3" * 64,
        labels_sha256="4" * 64,
        features_sha256="5" * 64,
        configuration_sha256="6" * 64,
    )


def _patch_validation(
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    policy_config: PolicyConfig,
    batch: PolicyValidationData,
) -> None:
    _patch_common(monkeypatch, modeling_config, policy_config)
    monkeypatch.setattr(policy_ops, "load_policy_validation_data", lambda *_: batch)
    monkeypatch.setattr(
        policy_ops, "portable_probabilities", lambda _m, _f: _probabilities(batch.targets)
    )


def test_validate_policy_publishes_an_honest_comparison_without_granting_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    policy_config: PolicyConfig,
) -> None:
    batch = _validation_batch(9_000, 180)
    _patch_validation(monkeypatch, modeling_config, policy_config, batch)
    directory = _published_band(tmp_path, policy_config)

    _, comparison = validate_policy(
        directory,
        Path("vsim"),
        Path("vfeat"),
        Path("config.json"),
        Path("model"),
        Path("policy.json"),
        tmp_path / "comparison",
        approve=False,
    )
    assert comparison["approval"]["approval_requested"] is False
    assert comparison["approval"]["approval_granted"] is False
    assert comparison["validation_batch"]["dataset_profile"] == "policy-validation"
    assert comparison["validation_batch"]["row_count"] == 9_000
    assert comparison["incumbent_policy"]["decision_threshold"] == INCUMBENT_THRESHOLD
    assert comparison["gates"]["observed_validation_attacks"] == 180
    rendered = str(comparison)
    for token in ("evt_", "cmp_", "scenario_type", "campaign_id"):
        assert token not in rendered


def test_validate_policy_refuses_approval_when_the_batch_is_too_small(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    policy_config: PolicyConfig,
) -> None:
    batch = _validation_batch(300, 30)
    _patch_validation(monkeypatch, modeling_config, policy_config, batch)
    directory = _published_band(tmp_path, policy_config)

    _, comparison = validate_policy(
        directory,
        Path("vsim"),
        Path("vfeat"),
        Path("config.json"),
        Path("model"),
        Path("policy.json"),
        tmp_path / "comparison",
        approve=True,
    )
    assert comparison["approval"]["approval_requested"] is True
    assert comparison["approval"]["approval_granted"] is False
    assert "validation_batch_below_minimum_evidence" in comparison["approval"]["refusal_reasons"]


def test_validate_policy_refuses_a_band_fitted_for_a_different_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    policy_config: PolicyConfig,
) -> None:
    batch = _validation_batch(9_000, 180)
    _patch_validation(monkeypatch, modeling_config, policy_config, batch)
    directory = _published_band(tmp_path, policy_config, model_id="b" * 64)
    with pytest.raises(PolicyOperationError, match="band_model_mismatch"):
        validate_policy(
            directory,
            Path("vsim"),
            Path("vfeat"),
            Path("config.json"),
            Path("model"),
            Path("policy.json"),
            tmp_path / "comparison",
            approve=False,
        )


def test_policy_file_metadata_hashes_real_bytes(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(b'{"a":1}\n')
    metadata = file_metadata(path)
    assert metadata["byte_size"] == 8
    assert isinstance(metadata["sha256"], str) and len(metadata["sha256"]) == 64
    with pytest.raises(PolicyArtifactError, match="source_unreadable"):
        file_metadata(tmp_path / "absent.json")
