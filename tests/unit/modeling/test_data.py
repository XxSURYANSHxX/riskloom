from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import riskloom.modeling.data as data_module
from riskloom.features.schema import FeatureRecord, FeatureVector
from riskloom.modeling.canonical import canonical_json_bytes
from riskloom.modeling.config import ModelingConfig
from riskloom.modeling.data import (
    EvaluationData,
    ModelingDataError,
    TrainingData,
    ValidationData,
    _validate_source_contract,
    load_evaluation_data,
    load_training_data,
)
from riskloom.simulation.config import (
    configuration_fingerprint,
    effective_configuration,
    load_generator_config,
)
from riskloom.simulation.label_schema import GroundTruthLabel


def _feature_vector() -> FeatureVector:
    values = {name: 0 for name in FeatureVector.model_fields}
    values.update({"amount_subunits": 100, "channel_web": 1})
    return FeatureVector.model_validate(values, strict=True)


def _row(
    index: int,
    split: str,
    occurred_at: datetime,
    *,
    attack: bool,
    campaign_id: str | None = None,
) -> tuple[bytes, bytes]:
    event_id = f"evt_{index:032x}"
    feature = FeatureRecord(
        event_id=event_id,
        occurred_at=occurred_at,
        features=_feature_vector(),
    )
    label = GroundTruthLabel.model_validate(
        {
            "campaign_id": campaign_id,
            "event_id": event_id,
            "generator_metadata": {
                "campaign_profile": "baseline_reuse" if attack else None,
                "component_version": "1.0.0",
                "scenario_instance_id": f"scn_{index:032x}",
            },
            "is_attack": attack,
            "scenario_type": "card_testing_campaign" if attack else "normal",
            "split": split,
        }
    )
    return (
        canonical_json_bytes(feature.model_dump(mode="json")),
        canonical_json_bytes(label.model_dump(mode="json")),
    )


def _tiny_sources(tmp_path: Path) -> tuple[Path, Path, datetime]:
    boundary = datetime(2026, 1, 2, tzinfo=UTC)
    rows = [
        _row(1, "train", boundary - timedelta(days=2), attack=False),
        _row(
            2,
            "train",
            boundary - timedelta(days=1, seconds=1),
            attack=True,
            campaign_id="cmp_" + "1" * 32,
        ),
        _row(3, "calibration", boundary - timedelta(seconds=2), attack=False),
        _row(
            4,
            "calibration",
            boundary - timedelta(seconds=1),
            attack=True,
            campaign_id="cmp_" + "2" * 32,
        ),
        _row(5, "calibration", boundary, attack=True, campaign_id="cmp_" + "3" * 32),
        _row(6, "calibration", boundary + timedelta(seconds=1), attack=False),
        _row(7, "test", boundary + timedelta(days=1), attack=False),
        _row(
            8,
            "test",
            boundary + timedelta(days=2),
            attack=True,
            campaign_id="cmp_" + "4" * 32,
        ),
    ]
    features = tmp_path / "features.jsonl"
    labels = tmp_path / "labels.jsonl"
    features.write_bytes(b"".join(row[0] for row in rows))
    labels.write_bytes(b"".join(row[1] for row in rows))
    return labels, features, boundary


def _patch_tiny_contract(
    monkeypatch: pytest.MonkeyPatch,
    labels: Path,
    features: Path,
    boundary: datetime,
    config: ModelingConfig,
) -> None:
    monkeypatch.setattr(
        data_module,
        "_validate_source_contract",
        lambda *_: (labels, features, boundary),
    )
    monkeypatch.setattr(data_module, "EXPECTED_TOTAL_ROWS", 8)
    monkeypatch.setattr(data_module, "EXPECTED_TRAIN_ROWS", 2)
    monkeypatch.setattr(data_module, "EXPECTED_CALIBRATION_FIT_ROWS", 2)
    monkeypatch.setattr(data_module, "EXPECTED_POLICY_SELECTION_ROWS", 2)
    monkeypatch.setattr(data_module, "EXPECTED_HELD_OUT_ROWS", 2)
    monkeypatch.setattr(data_module, "EXPECTED_CALIBRATION_CAMPAIGNS", 2)
    monkeypatch.setattr(data_module, "EXPECTED_ATTACKS_PER_CALIBRATION_CAMPAIGN", 1)
    monkeypatch.setattr(data_module, "EXPECTED_CAMPAIGNS_PER_CALIBRATION_SIDE", 1)
    monkeypatch.setattr(
        data_module,
        "file_metadata",
        lambda path: {
            "byte_size": path.stat().st_size,
            "sha256": (
                config.source_contract.labels_sha256
                if path == labels
                else config.source_contract.features_sha256
            ),
        },
    )


def _source_manifests(config: ModelingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    generator = load_generator_config(Path("configs/simulation/development.json"))
    contract = config.source_contract
    simulation = {
        "artifact_type": "synthetic_checkout_simulation",
        "artifacts": {
            "events.jsonl": {"sha256": contract.events_sha256},
            "labels.jsonl": {"sha256": contract.labels_sha256},
            "report.json": {"sha256": contract.simulation_report_sha256},
        },
        "config_schema_version": contract.simulation_config_schema_version,
        "dataset_id": contract.simulation_dataset_id,
        "effective_configuration": effective_configuration(generator),
        "effective_configuration_sha256": configuration_fingerprint(generator),
        "generator_version": contract.simulation_generator_version,
        "product": "RiskLoom",
        "splits": {
            "calibration": {
                "end_exclusive": "2026-01-26T00:00:00.000Z",
                "start_inclusive": "2026-01-21T00:00:00.000Z",
            }
        },
    }
    feature = {
        "artifact_type": "temporal_coordination_features",
        "artifacts": {
            "features.jsonl": {"sha256": contract.features_sha256},
            "report.json": {"sha256": contract.feature_report_sha256},
        },
        "feature_count": 75,
        "feature_dataset_id": contract.feature_dataset_id,
        "feature_engine_version": contract.feature_engine_version,
        "feature_schema_version": contract.feature_schema_version,
        "product": "RiskLoom",
        "source_events": {"sha256": contract.events_sha256},
    }
    return simulation, feature


def test_source_contract_recomputes_configuration_and_boundary(
    monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    simulation, feature = _source_manifests(modeling_config)
    monkeypatch.setattr(
        data_module,
        "_expect_manifest_hash",
        lambda path, _expected: simulation if path.parent.name == "simulation" else feature,
    )
    monkeypatch.setattr(data_module, "_expect_hash", lambda *_: None)
    labels, features, boundary = _validate_source_contract(
        Path("simulation"), Path("feature"), modeling_config
    )
    assert labels == Path("simulation/labels.jsonl")
    assert features == Path("feature/features.jsonl")
    assert boundary == datetime(2026, 1, 24, tzinfo=UTC)


@pytest.mark.parametrize(
    ("manifest_name", "field", "value", "message"),
    [
        ("simulation", "dataset_id", "wrong", "simulation_dataset_id"),
        ("feature", "feature_count", 74, "feature_count"),
        ("feature", "source_events", {}, "feature_source"),
    ],
)
def test_source_contract_fails_closed_on_manifest_tampering(
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    manifest_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    simulation, feature = _source_manifests(modeling_config)
    selected = simulation if manifest_name == "simulation" else feature
    selected[field] = value
    monkeypatch.setattr(
        data_module,
        "_expect_manifest_hash",
        lambda path, _expected: simulation if path.parent.name == "simulation" else feature,
    )
    monkeypatch.setattr(data_module, "_expect_hash", lambda *_: None)
    with pytest.raises(ModelingDataError, match=message):
        _validate_source_contract(Path("simulation"), Path("feature"), modeling_config)


def test_source_contract_rejects_declared_artifact_hash_tampering(
    monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    simulation, feature = _source_manifests(modeling_config)
    simulation["artifacts"]["labels.jsonl"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        data_module,
        "_expect_manifest_hash",
        lambda path, _expected: simulation if path.parent.name == "simulation" else feature,
    )
    monkeypatch.setattr(data_module, "_expect_hash", lambda *_: None)
    with pytest.raises(ModelingDataError, match="declared_hash"):
        _validate_source_contract(Path("simulation"), Path("feature"), modeling_config)


def test_training_loader_uses_exact_boundary_and_discards_held_out_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    loaded = load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=False)
    assert isinstance(loaded, TrainingData)
    assert loaded.calibration_fit.row_count == 2
    assert loaded.policy_selection.row_count == 2
    assert not hasattr(loaded, "test")


def test_validation_loader_samples_held_out_features_only_after_lock_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    loaded = load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=True)
    assert isinstance(loaded, ValidationData)
    assert loaded.held_out_feature_sample.shape == (2, 75)


def test_evaluation_loader_defers_held_out_class_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    loaded = load_evaluation_data(tmp_path, tmp_path, modeling_config)
    assert isinstance(loaded, EvaluationData)
    assert loaded.targets.tolist() == [0, 1]


def test_evaluation_loader_preserves_specific_duplicate_event_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    feature_rows = features.read_bytes().splitlines(keepends=True)
    label_rows = labels.read_bytes().splitlines(keepends=True)
    for rows in (feature_rows, label_rows):
        rows[-1] = rows[-1].replace(
            b"evt_00000000000000000000000000000008",
            b"evt_00000000000000000000000000000007",
        )
    features.write_bytes(b"".join(feature_rows))
    labels.write_bytes(b"".join(label_rows))
    with pytest.raises(ModelingDataError, match="^modeling_duplicate_event_id$"):
        load_evaluation_data(tmp_path, tmp_path, modeling_config)


def test_evaluation_loader_preserves_specific_join_and_order_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    labels.write_bytes(
        labels.read_bytes().replace(
            b"evt_00000000000000000000000000000001",
            b"evt_ffffffffffffffffffffffffffffffff",
            1,
        )
    )
    with pytest.raises(ModelingDataError, match="^modeling_event_join_mismatch$"):
        load_evaluation_data(tmp_path, tmp_path, modeling_config)

    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    feature_rows = features.read_bytes().splitlines(keepends=True)
    label_rows = labels.read_bytes().splitlines(keepends=True)
    feature_rows[0], feature_rows[1] = feature_rows[1], feature_rows[0]
    label_rows[0], label_rows[1] = label_rows[1], label_rows[0]
    features.write_bytes(b"".join(feature_rows))
    labels.write_bytes(b"".join(label_rows))
    with pytest.raises(ModelingDataError, match="^modeling_source_order_invalid$"):
        load_evaluation_data(tmp_path, tmp_path, modeling_config)


def test_evaluation_loader_preserves_noncanonical_row_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    feature_rows = features.read_bytes().splitlines(keepends=True)
    feature_rows[0] = feature_rows[0].replace(b'{"event_id"', b'{ "event_id"', 1)
    features.write_bytes(b"".join(feature_rows))
    with pytest.raises(ModelingDataError, match="^modeling_jsonl_not_canonical$"):
        load_evaluation_data(tmp_path, tmp_path, modeling_config)


def test_evaluation_loader_rejects_early_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    labels.write_bytes(b"".join(labels.read_bytes().splitlines(keepends=True)[:-1]))
    with pytest.raises(ModelingDataError, match="^modeling_source_early_eof$"):
        load_evaluation_data(tmp_path, tmp_path, modeling_config)


def test_evaluation_loader_rejects_wrong_held_out_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    monkeypatch.setattr(data_module, "EXPECTED_HELD_OUT_ROWS", 3)
    with pytest.raises(ModelingDataError, match="^modeling_evaluation_held_out_row_count_invalid$"):
        load_evaluation_data(tmp_path, tmp_path, modeling_config)


def test_training_loader_rejects_join_mismatch_and_noncanonical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    raw = labels.read_bytes().replace(
        b"evt_00000000000000000000000000000001", b"evt_ffffffffffffffffffffffffffffffff", 1
    )
    labels.write_bytes(raw)
    with pytest.raises(ModelingDataError, match="join_mismatch"):
        load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=False)


def test_training_loader_rejects_early_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    labels.write_bytes(b"".join(labels.read_bytes().splitlines(keepends=True)[:-1]))
    with pytest.raises(ModelingDataError, match="early_eof"):
        load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=False)


def test_training_loader_rejects_duplicate_and_reordered_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    feature_rows = features.read_bytes().splitlines(keepends=True)
    label_rows = labels.read_bytes().splitlines(keepends=True)
    feature_rows[-1] = feature_rows[-1].replace(
        b"evt_00000000000000000000000000000008",
        b"evt_00000000000000000000000000000007",
    )
    label_rows[-1] = label_rows[-1].replace(
        b"evt_00000000000000000000000000000008",
        b"evt_00000000000000000000000000000007",
    )
    features.write_bytes(b"".join(feature_rows))
    labels.write_bytes(b"".join(label_rows))
    with pytest.raises(ModelingDataError, match="duplicate_event_id"):
        load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=False)

    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    feature_rows = features.read_bytes().splitlines(keepends=True)
    label_rows = labels.read_bytes().splitlines(keepends=True)
    feature_rows[0], feature_rows[1] = feature_rows[1], feature_rows[0]
    label_rows[0], label_rows[1] = label_rows[1], label_rows[0]
    features.write_bytes(b"".join(feature_rows))
    labels.write_bytes(b"".join(label_rows))
    with pytest.raises(ModelingDataError, match="order_invalid"):
        load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=False)


def test_training_loader_rejects_campaign_crossing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modeling_config: ModelingConfig
) -> None:
    labels, features, boundary = _tiny_sources(tmp_path)
    _patch_tiny_contract(monkeypatch, labels, features, boundary, modeling_config)
    labels.write_bytes(
        labels.read_bytes().replace(
            b"cmp_33333333333333333333333333333333",
            b"cmp_22222222222222222222222222222222",
        )
    )
    monkeypatch.setattr(data_module, "EXPECTED_CALIBRATION_CAMPAIGNS", 1)
    monkeypatch.setattr(data_module, "EXPECTED_ATTACKS_PER_CALIBRATION_CAMPAIGN", 2)
    with pytest.raises(ModelingDataError, match="crosses_boundary"):
        load_training_data(tmp_path, tmp_path, modeling_config, include_held_out_sample=False)
