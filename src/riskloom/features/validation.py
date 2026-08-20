import asyncio
import json
import re
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import ValidationError

from riskloom.features.artifacts import (
    FEATURE_ARTIFACT_FILENAMES,
    FeatureArtifactError,
    FileMetadata,
    artifact_metadata,
    build_feature_manifest,
    canonical_json_bytes,
    feature_dataset_id,
    file_metadata,
)
from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine
from riskloom.features.reporting import FeatureStatistics, build_feature_report
from riskloom.features.schema import FEATURE_COUNT, FeatureRecord
from riskloom.simulation.event_schema import CheckoutAttemptEvent
from riskloom.simulation.replay import (
    ReplayConsumerError,
    ReplayInputError,
    ReplayOptions,
    replay_jsonl,
)

MAXIMUM_FEATURE_LINE_BYTES = 65_536
MAXIMUM_REPORT_BYTES = 1_048_576
MAXIMUM_MANIFEST_BYTES = 1_048_576


class FeatureValidationError(ValueError):
    """Safe streaming validation error without artifact contents."""


def _read_canonical_object(path: Path, maximum_bytes: int) -> dict[str, Any]:
    try:
        if path.stat().st_size > maximum_bytes:
            raise FeatureValidationError("feature_metadata_oversized")
        raw = path.read_bytes()
        if not raw.endswith(b"\n") or b"\r" in raw:
            raise FeatureValidationError("feature_artifact_not_canonical")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise FeatureValidationError("feature_metadata_schema_invalid")
        if canonical_json_bytes(value) != raw:
            raise FeatureValidationError("feature_artifact_not_canonical")
        return value
    except FeatureValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, FeatureArtifactError):
        raise FeatureValidationError("feature_metadata_unreadable") from None


class _LockstepConsumer:
    def __init__(self, stream: BinaryIO, config: FeatureConfig) -> None:
        self.stream = stream
        self.engine = FeatureEngine(config)
        self.statistics = FeatureStatistics()
        self.error: FeatureValidationError | None = None
        self.line_number = 0

    async def consume(self, event: CheckoutAttemptEvent) -> None:
        self.line_number += 1
        try:
            raw_line = self.stream.readline(MAXIMUM_FEATURE_LINE_BYTES + 1)
            if not raw_line:
                raise FeatureValidationError("feature_rows_early_eof")
            if len(raw_line) > MAXIMUM_FEATURE_LINE_BYTES:
                raise FeatureValidationError("feature_line_oversized")
            if not raw_line.endswith(b"\n") or b"\r" in raw_line or not raw_line.strip():
                raise FeatureValidationError("feature_artifact_not_canonical")
            try:
                actual = FeatureRecord.model_validate_json(raw_line)
            except (ValidationError, ValueError):
                raise FeatureValidationError("feature_record_schema_invalid") from None
            if canonical_json_bytes(actual.model_dump(mode="json")) != raw_line:
                raise FeatureValidationError("feature_artifact_not_canonical")
            expected = self.engine.process(event)
            if actual != expected:
                raise FeatureValidationError("feature_vector_mismatch")
            self.statistics.add(actual)
        except FeatureValidationError as error:
            self.error = error
            raise


async def _validate_lockstep(
    events_path: Path,
    features_path: Path,
    config: FeatureConfig,
) -> tuple[FeatureStatistics, dict[str, object]]:
    try:
        stream = features_path.open("rb")
    except OSError:
        raise FeatureValidationError("feature_rows_unreadable") from None
    with stream:
        consumer = _LockstepConsumer(stream, config)
        try:
            await replay_jsonl(events_path, consumer, ReplayOptions())
        except ReplayInputError as error:
            raise FeatureValidationError(str(error)) from None
        except ReplayConsumerError:
            if consumer.error is not None:
                raise consumer.error from None
            raise FeatureValidationError("feature_lockstep_validation_failed") from None
        if stream.read(1):
            raise FeatureValidationError("feature_rows_extra")
    if consumer.statistics.row_count == 0:
        raise FeatureValidationError("feature_source_empty")
    return consumer.statistics, consumer.engine.diagnostics()


def _validate_manifest_header(
    manifest: dict[str, Any], config: FeatureConfig, source: FileMetadata
) -> tuple[str, int, dict[str, Any]]:
    if manifest.get("product") != "RiskLoom" or manifest.get("artifact_type") != (
        "temporal_coordination_features"
    ):
        raise FeatureValidationError("feature_manifest_marker_invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"features.jsonl", "report.json"}:
        raise FeatureValidationError("feature_manifest_artifact_set_invalid")
    if "manifest.json" in artifacts:
        raise FeatureValidationError("feature_manifest_must_not_hash_itself")
    dataset_id = manifest.get("feature_dataset_id")
    row_count = manifest.get("row_count")
    if (
        not isinstance(dataset_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset_id) is None
        or type(row_count) is not int
        or row_count <= 0
    ):
        raise FeatureValidationError("feature_manifest_schema_invalid")
    expected_id = feature_dataset_id(config, source.sha256)
    if dataset_id != expected_id:
        raise FeatureValidationError("feature_dataset_identity_mismatch")
    if manifest.get("source_events") != source.as_dict():
        raise FeatureValidationError("feature_source_metadata_mismatch")
    return dataset_id, row_count, artifacts


def validate_feature_dataset(
    events_path: Path,
    config: FeatureConfig,
    feature_directory: Path,
) -> dict[str, int | str]:
    try:
        files = {path.name for path in feature_directory.iterdir()}
    except OSError:
        raise FeatureValidationError("feature_directory_unreadable") from None
    if files != set(FEATURE_ARTIFACT_FILENAMES):
        raise FeatureValidationError("feature_artifact_set_invalid")

    source_before = file_metadata(events_path)
    manifest = _read_canonical_object(feature_directory / "manifest.json", MAXIMUM_MANIFEST_BYTES)
    report = _read_canonical_object(feature_directory / "report.json", MAXIMUM_REPORT_BYTES)
    dataset_id, declared_rows, declared_artifacts = _validate_manifest_header(
        manifest, config, source_before
    )
    statistics, diagnostics = asyncio.run(
        _validate_lockstep(events_path, feature_directory / "features.jsonl", config)
    )
    source_after = file_metadata(events_path)
    if source_after != source_before:
        raise FeatureValidationError("source_events_changed_during_validation")
    if statistics.row_count != declared_rows:
        raise FeatureValidationError("feature_row_count_mismatch")
    expected_report = build_feature_report(dataset_id, statistics, diagnostics)
    if report != expected_report:
        raise FeatureValidationError("feature_report_recomputation_mismatch")
    actual_artifacts = {
        "features.jsonl": artifact_metadata(
            feature_directory / "features.jsonl", statistics.row_count
        ),
        "report.json": artifact_metadata(feature_directory / "report.json", 1),
    }
    if declared_artifacts != actual_artifacts:
        raise FeatureValidationError("feature_artifact_metadata_mismatch")
    expected_manifest = build_feature_manifest(
        config, dataset_id, source_before, actual_artifacts, statistics.row_count
    )
    if manifest != expected_manifest:
        raise FeatureValidationError("feature_manifest_schema_invalid")
    return {
        "feature_count": FEATURE_COUNT,
        "feature_dataset_id": dataset_id,
        "row_count": statistics.row_count,
        "status": "valid",
    }
