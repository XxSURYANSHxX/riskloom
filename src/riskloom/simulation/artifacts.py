import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from riskloom.simulation.config import GENERATOR_VERSION, GeneratorConfig
from riskloom.simulation.event_schema import CheckoutAttemptEvent
from riskloom.simulation.generation import GeneratedRecord, generate_records
from riskloom.simulation.label_schema import GroundTruthLabel
from riskloom.simulation.reporting import build_report


class ArtifactPublishError(ValueError):
    """Safe generation or publication error."""


def _canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (_canonical_dumps(value) + "\n").encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((_canonical_dumps(dict(row)) + "\n").encode("utf-8") for row in rows)


def write_canonical_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_event_jsonl(path: Path, events: Iterable[CheckoutAttemptEvent]) -> None:
    def rows() -> Iterable[Mapping[str, Any]]:
        for event in events:
            if type(event) is not CheckoutAttemptEvent:
                raise ArtifactPublishError("event_model_invalid")
            yield event.model_dump(mode="json")

    path.write_bytes(_canonical_jsonl_bytes(rows()))


def _write_label_jsonl(path: Path, labels: Iterable[GroundTruthLabel]) -> None:
    def rows() -> Iterable[Mapping[str, Any]]:
        for label in labels:
            if type(label) is not GroundTruthLabel:
                raise ArtifactPublishError("label_model_invalid")
            yield label.model_dump(mode="json")

    path.write_bytes(_canonical_jsonl_bytes(rows()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


ARTIFACT_FILENAMES = ("events.jsonl", "labels.jsonl", "report.json", "manifest.json")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    dataset_id: str
    event_count: int
    output_directory: Path
    artifact_hashes: dict[str, str]


def _timestamp(value: datetime) -> str:
    utc = value.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1_000:03d}Z"


def _split_manifest(config: GeneratorConfig) -> dict[str, Any]:
    start = config.start_at.astimezone(UTC)
    result: dict[str, Any] = {}
    for split in config.splits:
        end = start + timedelta(days=split.duration_days)
        result[split.name.value] = {
            "end_exclusive": _timestamp(end),
            "event_count": split.event_count,
            "start_inclusive": _timestamp(start),
        }
        start = end
    return {key: result[key] for key in sorted(result)}


def _safe_output_path(output_directory: Path) -> Path:
    try:
        absolute = output_directory.absolute()
        if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
            raise ArtifactPublishError("unsafe_symlinked_output_directory")
        resolved = output_directory.resolve(strict=False)
    except (OSError, RuntimeError):
        raise ArtifactPublishError("unsafe_output_directory") from None
    repository_root = Path(__file__).resolve().parents[3]
    forbidden = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        repository_root,
        Path(resolved.anchor).resolve(),
    }
    if resolved in forbidden:
        raise ArtifactPublishError("unsafe_output_directory")
    return resolved


def _validate_existing_output(output_directory: Path, overwrite: bool) -> bool:
    if not output_directory.exists():
        return False
    if not output_directory.is_dir():
        raise ArtifactPublishError("output_path_not_directory")
    try:
        existing = {path.name for path in output_directory.iterdir()}
    except OSError:
        raise ArtifactPublishError("output_directory_unreadable") from None
    if not existing:
        return False
    if not overwrite:
        raise ArtifactPublishError("output_directory_not_empty")
    if existing != set(ARTIFACT_FILENAMES):
        raise ArtifactPublishError("output_directory_contains_unknown_files")
    try:
        manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ArtifactPublishError("existing_manifest_unreadable") from None
    if not isinstance(manifest, dict):
        raise ArtifactPublishError("existing_manifest_invalid")
    if manifest.get("product") != "RiskLoom" or manifest.get("artifact_type") != (
        "synthetic_checkout_simulation"
    ):
        raise ArtifactPublishError("existing_manifest_marker_invalid")
    return True


def _artifact_metadata(path: Path, row_count: int) -> dict[str, Any]:
    return {
        "byte_size": path.stat().st_size,
        "row_count": row_count,
        "sha256": sha256_file(path),
    }


def build_manifest(
    config: GeneratorConfig,
    seed: int,
    dataset_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_type": "synthetic_checkout_simulation",
        "artifacts": {key: dict(artifacts[key]) for key in sorted(artifacts)},
        "config_schema_version": config.config_schema_version,
        "dataset_id": dataset_id,
        "effective_configuration": config.model_dump(mode="json"),
        "event_schema_version": "1.0.0",
        "generator_version": GENERATOR_VERSION,
        "label_schema_version": "1.0.0",
        "product": "RiskLoom",
        "seed": seed,
        "splits": _split_manifest(config),
    }


def _write_staged_dataset(
    staging: Path,
    records: list[GeneratedRecord],
    config: GeneratorConfig,
    seed: int,
) -> tuple[str, dict[str, str]]:
    effective_configuration = config.model_dump(mode="json")
    identity_material = {
        "effective_configuration": effective_configuration,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
    }
    dataset_id = canonical_sha256(identity_material)
    write_event_jsonl(staging / "events.jsonl", (record.event for record in records))
    _write_label_jsonl(staging / "labels.jsonl", (record.label for record in records))
    write_canonical_json(staging / "report.json", build_report(records, dataset_id, config))
    artifacts = {
        "events.jsonl": _artifact_metadata(staging / "events.jsonl", len(records)),
        "labels.jsonl": _artifact_metadata(staging / "labels.jsonl", len(records)),
        "report.json": _artifact_metadata(staging / "report.json", 1),
    }
    manifest = build_manifest(config, seed, dataset_id, artifacts)
    write_canonical_json(staging / "manifest.json", manifest)
    return dataset_id, {key: str(value["sha256"]) for key, value in artifacts.items()}


def generate_dataset(
    config: GeneratorConfig,
    seed: int,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> GenerationResult:
    from riskloom.simulation.validation import validate_dataset_directory

    output = _safe_output_path(output_directory)
    existing_output = _validate_existing_output(output, overwrite)
    if existing_output:
        try:
            validate_dataset_directory(output)
        except ValueError:
            raise ArtifactPublishError("existing_dataset_invalid") from None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".riskloom-simulation-staging-", dir=output.parent))
    except OSError:
        raise ArtifactPublishError("artifact_staging_failed") from None
    try:
        records = generate_records(config, seed)
        dataset_id, hashes = _write_staged_dataset(staging, records, config, seed)
        validate_dataset_directory(staging)
        output.mkdir(parents=False, exist_ok=True)
        for filename in ARTIFACT_FILENAMES:
            os.replace(staging / filename, output / filename)
        validate_dataset_directory(output)
        return GenerationResult(
            dataset_id=dataset_id,
            event_count=len(records),
            output_directory=output,
            artifact_hashes=hashes,
        )
    except OSError:
        raise ArtifactPublishError("artifact_publication_failed") from None
    finally:
        if staging.exists():
            if staging.parent != output.parent or not staging.name.startswith(
                ".riskloom-simulation-staging-"
            ):
                raise ArtifactPublishError("unsafe_staging_cleanup_target")
            try:
                shutil.rmtree(staging)
            except OSError:
                raise ArtifactPublishError("artifact_staging_cleanup_failed") from None
