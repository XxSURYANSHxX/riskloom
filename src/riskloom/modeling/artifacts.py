import importlib.metadata
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from riskloom.features.schema import FEATURE_NAMES
from riskloom.modeling.canonical import (
    ModelingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    file_metadata,
    read_canonical_json,
)
from riskloom.modeling.config import ModelingConfig
from riskloom.modeling.model import LockedModel

MODEL_ARTIFACT_FILENAMES = ("model.json", "training_report.json", "manifest.json")
EVALUATION_ARTIFACT_FILENAMES = ("evaluation.json", "manifest.json")
MODEL_MANIFEST_KEYS = {
    "artifact_type",
    "artifacts",
    "dependencies",
    "effective_modeling_configuration",
    "effective_modeling_configuration_sha256",
    "feature_order_sha256",
    "model_id",
    "model_schema_version",
    "product",
    "source",
    "training_report_schema_version",
}


@dataclass(frozen=True, slots=True)
class PublicationResult:
    artifact_id: str
    output_directory: Path
    artifact_hashes: dict[str, str]


def _safe_output_path(output_directory: Path) -> Path:
    try:
        absolute = output_directory.absolute()
        if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
            raise ModelingArtifactError("unsafe_symlinked_modeling_output")
        resolved = output_directory.resolve(strict=False)
    except (OSError, RuntimeError):
        raise ModelingArtifactError("unsafe_modeling_output") from None
    repository_root = Path(__file__).resolve().parents[3]
    if resolved in {
        Path.cwd().resolve(),
        Path.home().resolve(),
        repository_root,
        Path(resolved.anchor).resolve(),
    }:
        raise ModelingArtifactError("unsafe_modeling_output")
    return resolved


def _require_empty_destination(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ModelingArtifactError("modeling_output_not_directory")
    try:
        if next(output.iterdir(), None) is not None:
            raise ModelingArtifactError("modeling_output_not_empty")
    except OSError:
        raise ModelingArtifactError("modeling_output_unreadable") from None


def require_output_separate(output_directory: Path, source_directories: tuple[Path, ...]) -> None:
    try:
        output = output_directory.resolve(strict=False)
        sources = tuple(source.resolve(strict=False) for source in source_directories)
    except (OSError, RuntimeError):
        raise ModelingArtifactError("unsafe_modeling_path_overlap") from None
    if any(
        output == source or output.is_relative_to(source) or source.is_relative_to(output)
        for source in sources
    ):
        raise ModelingArtifactError("unsafe_modeling_path_overlap")


def _write(path: Path, value: Any) -> None:
    try:
        path.write_bytes(canonical_json_bytes(value))
    except OSError:
        raise ModelingArtifactError("modeling_artifact_write_failed") from None


def _publish(
    output_directory: Path,
    payloads: dict[str, Any],
    ordered_filenames: tuple[str, ...],
) -> None:
    output = _safe_output_path(output_directory)
    _require_empty_destination(output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".riskloom-modeling-staging-", dir=output.parent))
    except OSError:
        raise ModelingArtifactError("modeling_staging_failed") from None
    try:
        for filename in ordered_filenames:
            _write(staging / filename, payloads[filename])
        output.mkdir(parents=False, exist_ok=True)
        for filename in ordered_filenames:
            os.replace(staging / filename, output / filename)
    except OSError:
        raise ModelingArtifactError("modeling_publication_failed") from None
    finally:
        if staging.exists():
            if staging.parent != output.parent or not staging.name.startswith(
                ".riskloom-modeling-staging-"
            ):
                raise ModelingArtifactError("unsafe_modeling_staging_cleanup")
            try:
                shutil.rmtree(staging)
            except OSError:
                raise ModelingArtifactError("modeling_staging_cleanup_failed") from None


def modeling_config_sha256(config: ModelingConfig) -> str:
    return canonical_sha256(config.model_dump(mode="json"))


def runtime_dependency_versions() -> dict[str, str]:
    return {
        "numpy": importlib.metadata.version("numpy"),
        "python": "3.11",
        "scikit-learn": importlib.metadata.version("scikit-learn"),
    }


def model_identity_material(
    model_without_id: dict[str, Any], config: ModelingConfig
) -> dict[str, Any]:
    return {
        "effective_modeling_configuration_sha256": modeling_config_sha256(config),
        "feature_order_sha256": canonical_sha256(list(FEATURE_NAMES)),
        "model": model_without_id,
        "source_feature_dataset_id": config.source_contract.feature_dataset_id,
        "source_simulation_dataset_id": config.source_contract.simulation_dataset_id,
    }


def build_model_manifest(
    model: LockedModel,
    config: ModelingConfig,
    model_metadata: dict[str, int | str],
    report_metadata: dict[str, int | str],
    dependency_versions: dict[str, str],
) -> dict[str, Any]:
    contract = config.source_contract
    return {
        "artifact_type": "locked_binary_risk_model",
        "artifacts": {
            "model.json": model_metadata,
            "training_report.json": report_metadata,
        },
        "dependencies": {key: dependency_versions[key] for key in sorted(dependency_versions)},
        "effective_modeling_configuration": config.model_dump(mode="json"),
        "effective_modeling_configuration_sha256": modeling_config_sha256(config),
        "feature_order_sha256": canonical_sha256(list(FEATURE_NAMES)),
        "model_id": model.model_id,
        "model_schema_version": model.model_schema_version,
        "product": "RiskLoom",
        "source": {
            "events_sha256": contract.events_sha256,
            "feature_dataset_id": contract.feature_dataset_id,
            "feature_manifest_sha256": contract.feature_manifest_sha256,
            "feature_report_sha256": contract.feature_report_sha256,
            "features_sha256": contract.features_sha256,
            "labels_sha256": contract.labels_sha256,
            "simulation_dataset_id": contract.simulation_dataset_id,
            "simulation_manifest_sha256": contract.simulation_manifest_sha256,
            "simulation_report_sha256": contract.simulation_report_sha256,
        },
        "training_report_schema_version": "1.0.0",
    }


def publish_model(
    output_directory: Path,
    model: LockedModel,
    report: dict[str, Any],
    config: ModelingConfig,
    dependency_versions: dict[str, str],
) -> PublicationResult:
    model_value = model.model_dump(mode="json")
    model_bytes = canonical_json_bytes(model_value)
    report_bytes = canonical_json_bytes(report)
    model_metadata: dict[str, int | str] = {
        "byte_size": len(model_bytes),
        "sha256": canonical_sha256(model_value),
    }
    report_metadata: dict[str, int | str] = {
        "byte_size": len(report_bytes),
        "sha256": canonical_sha256(report),
    }
    manifest = build_model_manifest(
        model, config, model_metadata, report_metadata, dependency_versions
    )
    _publish(
        output_directory,
        {
            "model.json": model_value,
            "training_report.json": report,
            "manifest.json": manifest,
        },
        MODEL_ARTIFACT_FILENAMES,
    )
    return PublicationResult(
        artifact_id=model.model_id,
        output_directory=output_directory.resolve(),
        artifact_hashes={
            "manifest.json": canonical_sha256(manifest),
            "model.json": str(model_metadata["sha256"]),
            "training_report.json": str(report_metadata["sha256"]),
        },
    )


def load_locked_model(
    model_directory: Path, config: ModelingConfig
) -> tuple[LockedModel, dict[str, Any], dict[str, Any]]:
    try:
        existing = {path.name for path in model_directory.iterdir()}
    except OSError:
        raise ModelingArtifactError("locked_model_directory_unreadable") from None
    if existing != set(MODEL_ARTIFACT_FILENAMES):
        raise ModelingArtifactError("locked_model_artifact_set_invalid")
    model_value = read_canonical_json(model_directory / "model.json")
    report = read_canonical_json(model_directory / "training_report.json")
    manifest = read_canonical_json(model_directory / "manifest.json")
    try:
        model = LockedModel.model_validate(model_value, strict=True)
    except ValidationError:
        raise ModelingArtifactError("locked_model_schema_invalid") from None
    if manifest.get("product") != "RiskLoom" or manifest.get("artifact_type") != (
        "locked_binary_risk_model"
    ):
        raise ModelingArtifactError("locked_model_manifest_marker_invalid")
    if set(manifest) != MODEL_MANIFEST_KEYS:
        raise ModelingArtifactError("locked_model_manifest_schema_invalid")
    if manifest.get("effective_modeling_configuration") != config.model_dump(mode="json"):
        raise ModelingArtifactError("locked_model_configuration_mismatch")
    if manifest.get("effective_modeling_configuration_sha256") != modeling_config_sha256(config):
        raise ModelingArtifactError("locked_model_configuration_hash_mismatch")
    if manifest.get("feature_order_sha256") != canonical_sha256(list(FEATURE_NAMES)):
        raise ModelingArtifactError("locked_model_feature_order_hash_mismatch")
    if manifest.get("dependencies") != runtime_dependency_versions():
        raise ModelingArtifactError("locked_model_dependency_mismatch")
    contract = config.source_contract
    expected_source = {
        "events_sha256": contract.events_sha256,
        "feature_dataset_id": contract.feature_dataset_id,
        "feature_manifest_sha256": contract.feature_manifest_sha256,
        "feature_report_sha256": contract.feature_report_sha256,
        "features_sha256": contract.features_sha256,
        "labels_sha256": contract.labels_sha256,
        "simulation_dataset_id": contract.simulation_dataset_id,
        "simulation_manifest_sha256": contract.simulation_manifest_sha256,
        "simulation_report_sha256": contract.simulation_report_sha256,
    }
    if manifest.get("source") != expected_source:
        raise ModelingArtifactError("locked_model_source_mismatch")
    if (
        manifest.get("model_schema_version") != model.model_schema_version
        or manifest.get("training_report_schema_version") != "1.0.0"
    ):
        raise ModelingArtifactError("locked_model_version_mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ModelingArtifactError("locked_model_manifest_artifacts_invalid")
    if set(artifacts) != {"model.json", "training_report.json"}:
        raise ModelingArtifactError("locked_model_manifest_artifacts_invalid")
    for filename in ("model.json", "training_report.json"):
        expected = artifacts.get(filename)
        if not isinstance(expected, dict):
            raise ModelingArtifactError("locked_model_manifest_artifacts_invalid")
        actual = file_metadata(model_directory / filename)
        if expected != actual:
            raise ModelingArtifactError("locked_model_artifact_hash_mismatch")
    without_id = dict(model_value)
    without_id.pop("model_id", None)
    expected_id = canonical_sha256(model_identity_material(without_id, config))
    if model.model_id != expected_id or manifest.get("model_id") != expected_id:
        raise ModelingArtifactError("locked_model_identity_mismatch")
    return model, report, manifest


def publish_evaluation(
    output_directory: Path,
    evaluation: dict[str, Any],
    model: LockedModel,
    config: ModelingConfig,
    locked_model_sha256: str,
    locked_model_manifest_sha256: str,
) -> PublicationResult:
    evaluation_id = canonical_sha256(
        {
            "evaluation": evaluation,
            "model_id": model.model_id,
            "source_feature_dataset_id": config.source_contract.feature_dataset_id,
            "source_simulation_dataset_id": config.source_contract.simulation_dataset_id,
        }
    )
    evaluation = {**evaluation, "evaluation_id": evaluation_id}
    evaluation_metadata = {
        "byte_size": len(canonical_json_bytes(evaluation)),
        "sha256": canonical_sha256(evaluation),
    }
    manifest = {
        "artifact_type": "held_out_model_evaluation",
        "artifacts": {"evaluation.json": evaluation_metadata},
        "evaluation_id": evaluation_id,
        "evaluation_schema_version": "1.0.0",
        "dependencies": runtime_dependency_versions(),
        "effective_modeling_configuration": config.model_dump(mode="json"),
        "effective_modeling_configuration_sha256": modeling_config_sha256(config),
        "feature_order_sha256": canonical_sha256(list(FEATURE_NAMES)),
        "locked_model_sha256": locked_model_sha256,
        "locked_model_manifest_sha256": locked_model_manifest_sha256,
        "model_id": model.model_id,
        "product": "RiskLoom",
        "source_feature_dataset_id": config.source_contract.feature_dataset_id,
        "source_features_sha256": config.source_contract.features_sha256,
        "source_labels_sha256": config.source_contract.labels_sha256,
        "source_simulation_dataset_id": config.source_contract.simulation_dataset_id,
    }
    _publish(
        output_directory,
        {"evaluation.json": evaluation, "manifest.json": manifest},
        EVALUATION_ARTIFACT_FILENAMES,
    )
    return PublicationResult(
        artifact_id=evaluation_id,
        output_directory=output_directory.resolve(),
        artifact_hashes={
            "evaluation.json": str(evaluation_metadata["sha256"]),
            "manifest.json": canonical_sha256(manifest),
        },
    )
