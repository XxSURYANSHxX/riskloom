"""Disjointness of the counterfactual batch, and the approval gate's refusal behaviour."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import riskloom.modeling.data as data_module
from riskloom.features.schema import FeatureRecord, FeatureVector
from riskloom.modeling.canonical import canonical_json_bytes
from riskloom.modeling.config import ModelingConfig, SourceContract
from riskloom.modeling.data import ModelingDataError, load_policy_validation_data
from riskloom.simulation.config import (
    configuration_fingerprint,
    effective_configuration,
    load_generator_config,
)
from riskloom.simulation.label_schema import GroundTruthLabel

POLICY_VALIDATION_CONFIG = Path("configs/simulation/policy-validation.json")
DEVELOPMENT_CONFIG = Path("configs/simulation/development.json")


def _feature_vector() -> FeatureVector:
    values = {name: 0 for name in FeatureVector.model_fields}
    values.update({"amount_subunits": 100, "channel_web": 1})
    return FeatureVector.model_validate(values, strict=True)


def _row(index: int, occurred_at: datetime, *, attack: bool) -> tuple[bytes, bytes]:
    event_id = f"evt_{index:032x}"
    feature = FeatureRecord(event_id=event_id, occurred_at=occurred_at, features=_feature_vector())
    label = GroundTruthLabel.model_validate(
        {
            "campaign_id": ("cmp_" + f"{index:032x}") if attack else None,
            "event_id": event_id,
            "generator_metadata": {
                "campaign_profile": "baseline_reuse" if attack else None,
                "component_version": "1.0.0",
                "scenario_instance_id": f"scn_{index:032x}",
            },
            "is_attack": attack,
            "scenario_type": "card_testing_campaign" if attack else "normal",
            "split": "train",
        }
    )
    return (
        canonical_json_bytes(feature.model_dump(mode="json")),
        canonical_json_bytes(label.model_dump(mode="json")),
    )


def _write_batch(tmp_path: Path) -> tuple[Path, Path, str, str]:
    """Materialise a tiny but structurally valid policy-validation batch."""

    simulation = tmp_path / "sim"
    features = tmp_path / "feat"
    simulation.mkdir()
    features.mkdir()
    start = datetime(2026, 3, 1, tzinfo=UTC)
    rows = [
        _row(1, start, attack=False),
        _row(2, start + timedelta(seconds=1), attack=True),
        _row(3, start + timedelta(seconds=2), attack=False),
        _row(4, start + timedelta(seconds=3), attack=True),
    ]
    (simulation / "labels.jsonl").write_bytes(b"".join(row[1] for row in rows))
    (features / "features.jsonl").write_bytes(b"".join(row[0] for row in rows))

    generator = load_generator_config(POLICY_VALIDATION_CONFIG)
    fingerprint = str(configuration_fingerprint(generator))
    labels_sha = data_module.file_metadata(simulation / "labels.jsonl")["sha256"]
    features_sha = data_module.file_metadata(features / "features.jsonl")["sha256"]
    events_sha = "e" * 64

    simulation_manifest: dict[str, Any] = {
        "artifact_type": "synthetic_checkout_simulation",
        "artifacts": {
            "events.jsonl": {"sha256": events_sha},
            "labels.jsonl": {"sha256": labels_sha},
            "report.json": {"sha256": "f" * 64},
        },
        "config_schema_version": "1.1.0",
        "dataset_id": "1" * 64,
        "effective_configuration": effective_configuration(generator),
        "effective_configuration_sha256": fingerprint,
        "product": "RiskLoom",
    }
    feature_manifest: dict[str, Any] = {
        "artifact_type": "temporal_coordination_features",
        "artifacts": {"features.jsonl": {"sha256": features_sha}},
        "feature_count": 75,
        "feature_dataset_id": "2" * 64,
        "feature_engine_version": "1.0.0",
        "feature_schema_version": "1.0.0",
        "product": "RiskLoom",
        "source_events": {"sha256": events_sha},
    }
    (simulation / "manifest.json").write_bytes(canonical_json_bytes(simulation_manifest))
    (features / "manifest.json").write_bytes(canonical_json_bytes(feature_manifest))
    return simulation, features, fingerprint, events_sha


@pytest.fixture(scope="session")
def contract(modeling_config: ModelingConfig) -> SourceContract:
    return modeling_config.source_contract


def test_policy_validation_configuration_fingerprint_differs_from_the_locked_development_one(
    contract: SourceContract,
) -> None:
    """Explicit inequality against committed configuration, independent of any file path."""

    policy_validation = load_generator_config(POLICY_VALIDATION_CONFIG)
    development = load_generator_config(DEVELOPMENT_CONFIG)
    policy_fingerprint = configuration_fingerprint(policy_validation)
    development_fingerprint = configuration_fingerprint(development)
    assert policy_validation.dataset_profile == "policy-validation"
    assert development.dataset_profile == "development"
    assert policy_fingerprint != development_fingerprint
    assert policy_fingerprint != contract.simulation_configuration_sha256
    assert development_fingerprint == contract.simulation_configuration_sha256


def test_policy_validation_window_is_chronologically_disjoint_from_development() -> None:
    policy_validation = load_generator_config(POLICY_VALIDATION_CONFIG)
    development = load_generator_config(DEVELOPMENT_CONFIG)

    def window(config: Any) -> tuple[datetime, datetime]:
        days = sum(split.duration_days for split in config.splits)
        return config.start_at, config.start_at + timedelta(days=days)

    development_start, development_end = window(development)
    policy_start, policy_end = window(policy_validation)
    # Entirely after, with no overlap in either direction.
    assert policy_start >= development_end
    assert not (policy_start < development_end and development_start < policy_end)


def test_loader_accepts_a_distinct_policy_validation_batch(
    tmp_path: Path, contract: SourceContract
) -> None:
    simulation, features, fingerprint, _ = _write_batch(tmp_path)
    data = load_policy_validation_data(simulation, features, contract)
    assert data.row_count == 4
    assert data.targets.tolist() == [0, 1, 0, 1]
    assert data.features.shape == (4, 75)
    assert data.configuration_sha256 == fingerprint
    assert data.simulation_dataset_id != contract.simulation_dataset_id
    assert data.feature_dataset_id != contract.feature_dataset_id


@pytest.mark.parametrize(
    "field",
    [
        "simulation_dataset_id",
        "feature_dataset_id",
        "events_sha256",
        "labels_sha256",
        "features_sha256",
        "configuration_sha256",
    ],
)
def test_loader_refuses_a_batch_that_reuses_any_locked_development_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: SourceContract, field: str
) -> None:
    """Each identity is forced to the locked development value one at a time.

    ``labels_sha256`` and ``features_sha256`` are also covered by the on-disk integrity check,
    which fires first because a tampered declared hash no longer matches the real file. That check
    is disabled for those two cases only, so what is asserted here is the reuse guard itself rather
    than an earlier guard that happens to catch the same edit.
    """

    simulation, features, _, _ = _write_batch(tmp_path)
    simulation_manifest = json.loads((simulation / "manifest.json").read_bytes())
    feature_manifest = json.loads((features / "manifest.json").read_bytes())

    if field == "simulation_dataset_id":
        simulation_manifest["dataset_id"] = contract.simulation_dataset_id
    elif field == "feature_dataset_id":
        feature_manifest["feature_dataset_id"] = contract.feature_dataset_id
    elif field == "events_sha256":
        simulation_manifest["artifacts"]["events.jsonl"]["sha256"] = contract.events_sha256
        feature_manifest["source_events"]["sha256"] = contract.events_sha256
    elif field == "labels_sha256":
        simulation_manifest["artifacts"]["labels.jsonl"]["sha256"] = contract.labels_sha256
        monkeypatch.setattr(data_module, "_expect_hash", lambda *_: None)
    elif field == "features_sha256":
        feature_manifest["artifacts"]["features.jsonl"]["sha256"] = contract.features_sha256
        monkeypatch.setattr(data_module, "_expect_hash", lambda *_: None)
        monkeypatch.setattr(
            data_module,
            "file_metadata",
            lambda path: (
                {"byte_size": 0, "sha256": contract.features_sha256}
                if path.name == "features.jsonl"
                else {"byte_size": 0, "sha256": contract.labels_sha256}
            ),
        )
    else:
        simulation_manifest["effective_configuration_sha256"] = (
            contract.simulation_configuration_sha256
        )
    (simulation / "manifest.json").write_bytes(canonical_json_bytes(simulation_manifest))
    (features / "manifest.json").write_bytes(canonical_json_bytes(feature_manifest))

    expected = (
        "policy_validation_configuration_fingerprint_mismatch"
        if field == "configuration_sha256"
        else f"policy_validation_reuses_development_{field}"
    )
    with pytest.raises(ModelingDataError, match=f"^{expected}$"):
        load_policy_validation_data(simulation, features, contract)


def test_loader_refuses_a_development_profile_batch(
    tmp_path: Path, contract: SourceContract
) -> None:
    simulation, features, _, _ = _write_batch(tmp_path)
    manifest = json.loads((simulation / "manifest.json").read_bytes())
    manifest["effective_configuration"] = effective_configuration(
        load_generator_config(DEVELOPMENT_CONFIG)
    )
    manifest["effective_configuration_sha256"] = contract.simulation_configuration_sha256
    (simulation / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ModelingDataError, match="policy_validation"):
        load_policy_validation_data(simulation, features, contract)


def test_loader_refuses_a_feature_file_that_does_not_match_its_manifest(
    tmp_path: Path, contract: SourceContract
) -> None:
    simulation, features, _, _ = _write_batch(tmp_path)
    path = features / "features.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ModelingDataError, match="hash_mismatch"):
        load_policy_validation_data(simulation, features, contract)
