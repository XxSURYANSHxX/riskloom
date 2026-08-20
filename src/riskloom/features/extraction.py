import asyncio
from pathlib import Path
from typing import BinaryIO

from riskloom.features.artifacts import (
    FEATURE_ARTIFACT_FILENAMES,
    ExtractionResult,
    FeatureArtifactError,
    artifact_metadata,
    build_feature_manifest,
    canonical_json_bytes,
    cleanup_staging_directory,
    create_staging_directory,
    feature_dataset_id,
    file_metadata,
    inspect_existing_output,
    publish_staged_artifacts,
    safe_output_path,
    write_canonical_json,
)
from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine
from riskloom.features.reporting import FeatureStatistics, build_feature_report
from riskloom.features.schema import FEATURE_COUNT
from riskloom.simulation.event_schema import CheckoutAttemptEvent
from riskloom.simulation.replay import (
    ReplayConsumerError,
    ReplayInputError,
    ReplayOptions,
    replay_jsonl,
)


class FeatureExtractionError(ValueError):
    """Safe feature extraction error."""


class _ExtractionConsumer:
    def __init__(
        self,
        engine: FeatureEngine,
        stream: BinaryIO,
        statistics: FeatureStatistics,
    ) -> None:
        self.engine = engine
        self.stream = stream
        self.statistics = statistics
        self.error: FeatureExtractionError | None = None

    async def consume(self, event: CheckoutAttemptEvent) -> None:
        record = self.engine.process(event)
        try:
            self.stream.write(canonical_json_bytes(record.model_dump(mode="json")))
        except (FeatureArtifactError, OSError):
            self.error = FeatureExtractionError("feature_artifact_write_failed")
            raise self.error from None
        self.statistics.add(record)


async def _extract_to_path(
    events_path: Path,
    features_path: Path,
    config: FeatureConfig,
) -> tuple[FeatureStatistics, dict[str, object]]:
    engine = FeatureEngine(config)
    statistics = FeatureStatistics()
    consumer: _ExtractionConsumer | None = None
    try:
        with features_path.open("wb") as stream:
            consumer = _ExtractionConsumer(engine, stream, statistics)
            await replay_jsonl(events_path, consumer, ReplayOptions())
    except ReplayInputError as error:
        raise FeatureExtractionError(str(error)) from None
    except ReplayConsumerError:
        if consumer is not None and consumer.error is not None:
            raise consumer.error from None
        raise FeatureExtractionError("feature_extraction_failed") from None
    except FeatureExtractionError:
        raise
    except OSError:
        raise FeatureExtractionError("feature_artifact_write_failed") from None
    except Exception:
        raise FeatureExtractionError("feature_extraction_failed") from None
    if statistics.row_count == 0:
        raise FeatureExtractionError("feature_source_empty")
    return statistics, engine.diagnostics()


def extract_feature_dataset(
    events_path: Path,
    config: FeatureConfig,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> ExtractionResult:
    from riskloom.features.validation import (
        FeatureValidationError,
        validate_feature_dataset,
    )

    output = safe_output_path(output_directory, events_path)
    existing = inspect_existing_output(output)
    if existing:
        if not overwrite:
            raise FeatureArtifactError("feature_output_directory_not_empty")
        if existing != set(FEATURE_ARTIFACT_FILENAMES):
            raise FeatureArtifactError("feature_output_contains_unknown_files")

    source_before = file_metadata(events_path)
    if existing:
        try:
            validate_feature_dataset(events_path, config, output)
        except FeatureValidationError:
            raise FeatureArtifactError("existing_feature_dataset_invalid") from None
        if file_metadata(events_path) != source_before:
            raise FeatureExtractionError("source_events_changed_during_existing_validation")

    dataset_id = feature_dataset_id(config, source_before.sha256)
    staging = create_staging_directory(output)
    try:
        statistics, diagnostics = asyncio.run(
            _extract_to_path(events_path, staging / "features.jsonl", config)
        )
        if file_metadata(events_path) != source_before:
            raise FeatureExtractionError("source_events_changed_during_extraction")
        report = build_feature_report(dataset_id, statistics, diagnostics)
        write_canonical_json(staging / "report.json", report)
        artifacts = {
            "features.jsonl": artifact_metadata(staging / "features.jsonl", statistics.row_count),
            "report.json": artifact_metadata(staging / "report.json", 1),
        }
        manifest = build_feature_manifest(
            config, dataset_id, source_before, artifacts, statistics.row_count
        )
        write_canonical_json(staging / "manifest.json", manifest)
        validate_feature_dataset(events_path, config, staging)
        if file_metadata(events_path) != source_before:
            raise FeatureExtractionError("source_events_changed_before_publication")
        publish_staged_artifacts(staging, output, replacing=bool(existing))
        validate_feature_dataset(events_path, config, output)
        return ExtractionResult(
            feature_dataset_id=dataset_id,
            row_count=statistics.row_count,
            feature_count=FEATURE_COUNT,
            output_directory=output,
            artifact_hashes={
                name: str(metadata["sha256"]) for name, metadata in sorted(artifacts.items())
            },
        )
    finally:
        cleanup_staging_directory(staging, output.parent)
