"""Canonical, manifest-last publication for policy band and comparison artifacts."""

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskloom.policy.canonical import (
    PolicyArtifactError,
    canonical_json_bytes,
    canonical_sha256,
)
from riskloom.policy.config import (
    POLICY_BAND_SCHEMA_VERSION,
    POLICY_COMPARISON_SCHEMA_VERSION,
    PolicyConfig,
)

BAND_ARTIFACT_FILENAMES = ("band.json", "manifest.json")
COMPARISON_ARTIFACT_FILENAMES = ("comparison.json", "manifest.json")


@dataclass(frozen=True, slots=True)
class PolicyPublicationResult:
    artifact_id: str
    output_directory: Path
    artifact_hashes: dict[str, str]


def policy_config_sha256(config: PolicyConfig) -> str:
    return canonical_sha256(config.model_dump(mode="json"))


def _safe_output_path(output_directory: Path) -> Path:
    try:
        absolute = output_directory.absolute()
        if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
            raise PolicyArtifactError("unsafe_symlinked_policy_output")
        resolved = output_directory.resolve(strict=False)
    except (OSError, RuntimeError):
        raise PolicyArtifactError("unsafe_policy_output") from None
    repository_root = Path(__file__).resolve().parents[3]
    if resolved in {
        Path.cwd().resolve(),
        Path.home().resolve(),
        repository_root,
        Path(resolved.anchor).resolve(),
    }:
        raise PolicyArtifactError("unsafe_policy_output")
    return resolved


def _require_empty_destination(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise PolicyArtifactError("policy_output_not_directory")
    try:
        if next(output.iterdir(), None) is not None:
            raise PolicyArtifactError("policy_output_not_empty")
    except OSError:
        raise PolicyArtifactError("policy_output_unreadable") from None


def require_output_separate(output_directory: Path, source_directories: tuple[Path, ...]) -> None:
    try:
        output = output_directory.resolve(strict=False)
        sources = tuple(source.resolve(strict=False) for source in source_directories)
    except (OSError, RuntimeError):
        raise PolicyArtifactError("unsafe_policy_path_overlap") from None
    if any(
        output == source or output.is_relative_to(source) or source.is_relative_to(output)
        for source in sources
    ):
        raise PolicyArtifactError("unsafe_policy_path_overlap")


def _publish(
    output_directory: Path, payloads: dict[str, Any], ordered_filenames: tuple[str, ...]
) -> None:
    output = _safe_output_path(output_directory)
    _require_empty_destination(output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".riskloom-policy-staging-", dir=output.parent))
    except OSError:
        raise PolicyArtifactError("policy_staging_failed") from None
    try:
        for filename in ordered_filenames:
            (staging / filename).write_bytes(canonical_json_bytes(payloads[filename]))
        output.mkdir(parents=False, exist_ok=True)
        for filename in ordered_filenames:
            os.replace(staging / filename, output / filename)
    except OSError:
        raise PolicyArtifactError("policy_publication_failed") from None
    finally:
        if staging.exists():
            if staging.parent != output.parent or not staging.name.startswith(
                ".riskloom-policy-staging-"
            ):
                raise PolicyArtifactError("unsafe_policy_staging_cleanup")
            try:
                shutil.rmtree(staging)
            except OSError:
                raise PolicyArtifactError("policy_staging_cleanup_failed") from None


def _artifact_metadata(value: Any) -> dict[str, int | str]:
    return {"byte_size": len(canonical_json_bytes(value)), "sha256": canonical_sha256(value)}


def publish_band(
    output_directory: Path,
    band_payload: dict[str, Any],
    config: PolicyConfig,
    source: dict[str, Any],
) -> PolicyPublicationResult:
    """Publish the fitted band. The manifest hashes band.json and never hashes itself."""

    band_id = canonical_sha256(
        {
            "band": band_payload,
            "effective_policy_configuration_sha256": policy_config_sha256(config),
            "source": source,
        }
    )
    band = {**band_payload, "band_id": band_id}
    manifest = {
        "artifact_type": "cost_aware_policy_band",
        "artifacts": {"band.json": _artifact_metadata(band)},
        "band_id": band_id,
        "effective_policy_configuration": config.model_dump(mode="json"),
        "effective_policy_configuration_sha256": policy_config_sha256(config),
        "policy_band_schema_version": POLICY_BAND_SCHEMA_VERSION,
        "product": "RiskLoom",
        "source": source,
    }
    _publish(
        output_directory, {"band.json": band, "manifest.json": manifest}, BAND_ARTIFACT_FILENAMES
    )
    return PolicyPublicationResult(
        artifact_id=band_id,
        output_directory=output_directory.resolve(),
        artifact_hashes={
            "band.json": canonical_sha256(band),
            "manifest.json": canonical_sha256(manifest),
        },
    )


def publish_comparison(
    output_directory: Path,
    comparison_payload: dict[str, Any],
    config: PolicyConfig,
    source: dict[str, Any],
) -> PolicyPublicationResult:
    """Publish the counterfactual comparison report.

    The report always records the gate outcome, including when the banded policy loses. Nothing in
    this function can mark a policy active; approval status is carried as data that a human step
    sets, and is only ever ``true`` when every gate passed.
    """

    comparison_id = canonical_sha256(
        {
            "comparison": comparison_payload,
            "effective_policy_configuration_sha256": policy_config_sha256(config),
            "source": source,
        }
    )
    comparison = {**comparison_payload, "comparison_id": comparison_id}
    manifest = {
        "artifact_type": "policy_counterfactual_comparison",
        "artifacts": {"comparison.json": _artifact_metadata(comparison)},
        "comparison_id": comparison_id,
        "effective_policy_configuration": config.model_dump(mode="json"),
        "effective_policy_configuration_sha256": policy_config_sha256(config),
        "policy_comparison_schema_version": POLICY_COMPARISON_SCHEMA_VERSION,
        "product": "RiskLoom",
        "source": source,
    }
    _publish(
        output_directory,
        {"comparison.json": comparison, "manifest.json": manifest},
        COMPARISON_ARTIFACT_FILENAMES,
    )
    return PolicyPublicationResult(
        artifact_id=comparison_id,
        output_directory=output_directory.resolve(),
        artifact_hashes={
            "comparison.json": canonical_sha256(comparison),
            "manifest.json": canonical_sha256(manifest),
        },
    )
