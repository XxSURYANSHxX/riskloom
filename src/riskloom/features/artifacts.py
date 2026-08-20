import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskloom.features.config import (
    FEATURE_CONFIG_SCHEMA_VERSION,
    FEATURE_ENGINE_VERSION,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
)
from riskloom.features.schema import FEATURE_COUNT

FEATURE_ARTIFACT_FILENAMES = ("features.jsonl", "report.json", "manifest.json")


class FeatureArtifactError(ValueError):
    """Safe feature-artifact or publication error."""


@dataclass(frozen=True, slots=True)
class FileMetadata:
    byte_size: int
    sha256: str

    def as_dict(self) -> dict[str, int | str]:
        return {"byte_size": self.byte_size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    feature_dataset_id: str
    row_count: int
    feature_count: int
    output_directory: Path
    artifact_hashes: dict[str, str]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise FeatureArtifactError("feature_artifact_not_canonicalizable") from None
    return (rendered + "\n").encode("utf-8")


def write_canonical_json(path: Path, value: Any) -> None:
    try:
        path.write_bytes(canonical_json_bytes(value))
    except OSError:
        raise FeatureArtifactError("feature_artifact_write_failed") from None


def file_metadata(path: Path) -> FileMetadata:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65_536), b""):
                digest.update(chunk)
                byte_size += len(chunk)
    except OSError:
        raise FeatureArtifactError("feature_file_unreadable") from None
    return FileMetadata(byte_size=byte_size, sha256=digest.hexdigest())


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def feature_dataset_id(config: FeatureConfig, source_sha256: str) -> str:
    return canonical_sha256(
        {
            "effective_configuration": config.model_dump(mode="json"),
            "feature_engine_version": FEATURE_ENGINE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_events_sha256": source_sha256,
        }
    )


def artifact_metadata(path: Path, row_count: int) -> dict[str, int | str]:
    metadata = file_metadata(path)
    return {
        "byte_size": metadata.byte_size,
        "row_count": row_count,
        "sha256": metadata.sha256,
    }


def build_feature_manifest(
    config: FeatureConfig,
    dataset_id: str,
    source_metadata: FileMetadata,
    artifacts: Mapping[str, Mapping[str, int | str]],
    row_count: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "temporal_coordination_features",
        "artifacts": {name: dict(artifacts[name]) for name in sorted(artifacts)},
        "effective_configuration": config.model_dump(mode="json"),
        "feature_config_schema_version": FEATURE_CONFIG_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "feature_dataset_id": dataset_id,
        "feature_engine_version": FEATURE_ENGINE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "product": "RiskLoom",
        "row_count": row_count,
        "source_events": source_metadata.as_dict(),
    }


def safe_output_path(output_directory: Path, source_path: Path) -> Path:
    try:
        absolute = output_directory.absolute()
        if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
            raise FeatureArtifactError("unsafe_symlinked_feature_output")
        resolved = output_directory.resolve(strict=False)
        source_resolved = source_path.resolve(strict=False)
    except (OSError, RuntimeError):
        raise FeatureArtifactError("unsafe_feature_output_directory") from None
    repository_root = Path(__file__).resolve().parents[3]
    forbidden = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        repository_root,
        Path(resolved.anchor).resolve(),
    }
    if resolved in forbidden:
        raise FeatureArtifactError("unsafe_feature_output_directory")
    if source_resolved == resolved or source_resolved.is_relative_to(resolved):
        raise FeatureArtifactError("unsafe_source_output_overlap")
    return resolved


def inspect_existing_output(output_directory: Path) -> set[str]:
    if not output_directory.exists():
        return set()
    if not output_directory.is_dir():
        raise FeatureArtifactError("feature_output_path_not_directory")
    try:
        return {path.name for path in output_directory.iterdir()}
    except OSError:
        raise FeatureArtifactError("feature_output_directory_unreadable") from None


def create_staging_directory(output_directory: Path) -> Path:
    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(prefix=".riskloom-features-staging-", dir=output_directory.parent)
        )
    except OSError:
        raise FeatureArtifactError("feature_artifact_staging_failed") from None


def publish_staged_artifacts(staging: Path, output: Path, replacing: bool) -> None:
    try:
        output.mkdir(parents=False, exist_ok=True)
        if replacing:
            (output / "manifest.json").unlink()
        for filename in ("features.jsonl", "report.json", "manifest.json"):
            os.replace(staging / filename, output / filename)
    except OSError:
        raise FeatureArtifactError("feature_artifact_publication_failed") from None


def cleanup_staging_directory(staging: Path, expected_parent: Path) -> None:
    if not staging.exists():
        return
    if staging.parent != expected_parent or not staging.name.startswith(
        ".riskloom-features-staging-"
    ):
        raise FeatureArtifactError("unsafe_feature_staging_cleanup_target")
    try:
        shutil.rmtree(staging)
    except OSError:
        raise FeatureArtifactError("feature_artifact_staging_cleanup_failed") from None
