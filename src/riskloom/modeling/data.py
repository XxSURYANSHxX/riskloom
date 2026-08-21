import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES, FeatureRecord
from riskloom.modeling.canonical import (
    ModelingArtifactError,
    file_metadata,
    parse_canonical_jsonl_row,
    read_canonical_json,
)
from riskloom.modeling.config import ModelingConfig
from riskloom.simulation.config import (
    GeneratorConfig,
    boundary_timestamp,
    configuration_fingerprint,
    validate_profile_contract,
)
from riskloom.simulation.label_schema import GroundTruthLabel, SplitName

EVENT_ID_PATTERN = re.compile(r"^evt_[0-9a-f]{32}$")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
EXPECTED_TOTAL_ROWS = 100_000
EXPECTED_TRAIN_ROWS = 66_000
EXPECTED_CALIBRATION_FIT_ROWS = 9_048
EXPECTED_POLICY_SELECTION_ROWS = 7_952
EXPECTED_HELD_OUT_ROWS = 17_000
EXPECTED_CALIBRATION_CAMPAIGNS = 10
EXPECTED_ATTACKS_PER_CALIBRATION_CAMPAIGN = 34
EXPECTED_CAMPAIGNS_PER_CALIBRATION_SIDE = 5


class ModelingDataError(ValueError):
    """A safe source-integrity or partitioning error."""


@dataclass(frozen=True, slots=True)
class PartitionData:
    features: np.ndarray
    targets: np.ndarray
    scenarios: tuple[str, ...]
    campaign_ids: tuple[str | None, ...]
    occurred_at_ms: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True, slots=True)
class TrainingData:
    train: PartitionData
    calibration_fit: PartitionData
    policy_selection: PartitionData
    boundary_timestamp: str


@dataclass(frozen=True, slots=True)
class ValidationData:
    training: TrainingData
    held_out_feature_sample: np.ndarray


@dataclass(frozen=True, slots=True)
class EvaluationData:
    features: np.ndarray
    targets: np.ndarray
    scenarios: tuple[str, ...]
    campaign_ids: tuple[str | None, ...]
    occurred_at_ms: np.ndarray


class _PartitionBuilder:
    def __init__(self) -> None:
        self.features: list[list[int]] = []
        self.targets: list[int] = []
        self.scenarios: list[str] = []
        self.campaign_ids: list[str | None] = []
        self.occurred_at_ms: list[int] = []

    def append(self, record: FeatureRecord, label: GroundTruthLabel) -> None:
        values = record.features.model_dump()
        self.features.append([values[name] for name in FEATURE_NAMES])
        self.targets.append(int(label.is_attack))
        self.scenarios.append(label.scenario_type.value)
        self.campaign_ids.append(label.campaign_id)
        self.occurred_at_ms.append(_timestamp_ms(record.occurred_at))

    def finish(self) -> PartitionData:
        array = np.asarray(self.features, dtype=np.float64)
        if array.size == 0:
            array = np.empty((0, FEATURE_COUNT), dtype=np.float64)
        return PartitionData(
            features=array,
            targets=np.asarray(self.targets, dtype=np.int8),
            scenarios=tuple(self.scenarios),
            campaign_ids=tuple(self.campaign_ids),
            occurred_at_ms=np.asarray(self.occurred_at_ms, dtype=np.int64),
        )


def _expect_hash(path: Path, expected: str) -> None:
    if file_metadata(path)["sha256"] != expected:
        raise ModelingDataError("modeling_source_hash_mismatch")


def _expect_manifest_hash(path: Path, expected: str) -> dict[str, object]:
    _expect_hash(path, expected)
    return read_canonical_json(path)


def _validate_source_contract(
    simulation_directory: Path,
    feature_directory: Path,
    config: ModelingConfig,
) -> tuple[Path, Path, datetime]:
    contract = config.source_contract
    simulation_manifest_path = simulation_directory / "manifest.json"
    feature_manifest_path = feature_directory / "manifest.json"
    simulation_manifest = _expect_manifest_hash(
        simulation_manifest_path, contract.simulation_manifest_sha256
    )
    feature_manifest = _expect_manifest_hash(
        feature_manifest_path, contract.feature_manifest_sha256
    )

    checks = (
        (simulation_manifest.get("product") == "RiskLoom", "simulation_product"),
        (
            simulation_manifest.get("artifact_type") == "synthetic_checkout_simulation",
            "simulation_artifact_type",
        ),
        (
            simulation_manifest.get("dataset_id") == contract.simulation_dataset_id,
            "simulation_dataset_id",
        ),
        (
            simulation_manifest.get("generator_version") == contract.simulation_generator_version,
            "simulation_generator_version",
        ),
        (
            simulation_manifest.get("config_schema_version")
            == contract.simulation_config_schema_version,
            "simulation_config_schema_version",
        ),
        (
            simulation_manifest.get("effective_configuration_sha256")
            == contract.simulation_configuration_sha256,
            "simulation_configuration_sha256",
        ),
        (feature_manifest.get("product") == "RiskLoom", "feature_product"),
        (
            feature_manifest.get("artifact_type") == "temporal_coordination_features",
            "feature_artifact_type",
        ),
        (
            feature_manifest.get("feature_dataset_id") == contract.feature_dataset_id,
            "feature_dataset_id",
        ),
        (
            feature_manifest.get("feature_engine_version") == contract.feature_engine_version,
            "feature_engine_version",
        ),
        (
            feature_manifest.get("feature_schema_version") == contract.feature_schema_version,
            "feature_schema_version",
        ),
        (feature_manifest.get("feature_count") == contract.feature_count, "feature_count"),
    )
    for valid, name in checks:
        if not valid:
            raise ModelingDataError(f"modeling_source_contract_{name}_mismatch")

    try:
        effective = simulation_manifest["effective_configuration"]
        generator_config = GeneratorConfig.model_validate(effective)
        validate_profile_contract(generator_config)
    except (KeyError, TypeError, ValidationError, ValueError):
        raise ModelingDataError("modeling_simulation_configuration_invalid") from None
    if generator_config.dataset_profile != "development":
        raise ModelingDataError("modeling_requires_development_profile")
    if configuration_fingerprint(generator_config) != contract.simulation_configuration_sha256:
        raise ModelingDataError("modeling_simulation_configuration_fingerprint_mismatch")

    simulation_artifacts = simulation_manifest.get("artifacts")
    feature_artifacts = feature_manifest.get("artifacts")
    if not isinstance(simulation_artifacts, dict) or not isinstance(feature_artifacts, dict):
        raise ModelingDataError("modeling_source_artifact_metadata_invalid")
    required_metadata = (
        (simulation_artifacts, "events.jsonl", contract.events_sha256),
        (simulation_artifacts, "labels.jsonl", contract.labels_sha256),
        (simulation_artifacts, "report.json", contract.simulation_report_sha256),
        (feature_artifacts, "features.jsonl", contract.features_sha256),
        (feature_artifacts, "report.json", contract.feature_report_sha256),
    )
    for artifacts, name, expected in required_metadata:
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict) or metadata.get("sha256") != expected:
            raise ModelingDataError("modeling_source_declared_hash_mismatch")
    source_events = feature_manifest.get("source_events")
    if not isinstance(source_events, dict) or source_events.get("sha256") != contract.events_sha256:
        raise ModelingDataError("modeling_feature_source_mismatch")

    labels_path = simulation_directory / "labels.jsonl"
    features_path = feature_directory / "features.jsonl"
    _expect_hash(labels_path, contract.labels_sha256)
    _expect_hash(simulation_directory / "report.json", contract.simulation_report_sha256)
    _expect_hash(features_path, contract.features_sha256)
    _expect_hash(feature_directory / "report.json", contract.feature_report_sha256)

    try:
        split_metadata = simulation_manifest["splits"]
        if not isinstance(split_metadata, dict):
            raise TypeError
        calibration = split_metadata["calibration"]
        if not isinstance(calibration, dict):
            raise TypeError
        start = _parse_timestamp(calibration["start_inclusive"])
        end = _parse_timestamp(calibration["end_exclusive"])
    except (KeyError, TypeError, ValueError):
        raise ModelingDataError("modeling_calibration_metadata_invalid") from None
    boundary = boundary_timestamp(start, end, config.calibration_boundary_basis_points)
    return labels_path, features_path, boundary


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("invalid timestamp")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("invalid UTC timestamp")
    return parsed


def _timestamp_ms(value: datetime) -> int:
    delta = value - EPOCH
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _validate_join_metadata(
    feature: dict[str, object], label: dict[str, object]
) -> tuple[str, str]:
    event_id = feature.get("event_id")
    if (
        not isinstance(event_id, str)
        or EVENT_ID_PATTERN.fullmatch(event_id) is None
        or label.get("event_id") != event_id
    ):
        raise ModelingDataError("modeling_event_join_mismatch")
    split = label.get("split")
    if split not in {member.value for member in SplitName}:
        raise ModelingDataError("modeling_split_invalid")
    occurred_at = feature.get("occurred_at")
    if not isinstance(occurred_at, str):
        raise ModelingDataError("modeling_feature_timestamp_invalid")
    return split, occurred_at


def _parse_training_row(
    feature_value: dict[str, object], label_value: dict[str, object]
) -> tuple[FeatureRecord, GroundTruthLabel]:
    try:
        feature = FeatureRecord.model_validate(feature_value, strict=False)
        label = GroundTruthLabel.model_validate(label_value, strict=False)
    except ValidationError:
        raise ModelingDataError("modeling_training_row_invalid") from None
    return feature, label


def _validate_campaign_partition(
    campaign_sides: dict[str, set[str]], campaign_counts: dict[str, int]
) -> None:
    if (
        len(campaign_counts) != EXPECTED_CALIBRATION_CAMPAIGNS
        or sorted(campaign_counts.values())
        != [EXPECTED_ATTACKS_PER_CALIBRATION_CAMPAIGN] * EXPECTED_CALIBRATION_CAMPAIGNS
    ):
        raise ModelingDataError("modeling_calibration_campaign_contract_invalid")
    if any(len(sides) != 1 for sides in campaign_sides.values()):
        raise ModelingDataError("modeling_calibration_campaign_crosses_boundary")
    side_counts = {"calibration_fit": 0, "policy_selection": 0}
    for sides in campaign_sides.values():
        side_counts[next(iter(sides))] += 1
    if side_counts != {
        "calibration_fit": EXPECTED_CAMPAIGNS_PER_CALIBRATION_SIDE,
        "policy_selection": EXPECTED_CAMPAIGNS_PER_CALIBRATION_SIDE,
    }:
        raise ModelingDataError("modeling_calibration_campaign_side_count_invalid")


def _sample_indices(total: int, maximum: int) -> set[int]:
    count = min(total, maximum)
    if count == 0:
        return set()
    if count == 1:
        return {0}
    return {(index * (total - 1)) // (count - 1) for index in range(count)}


def load_training_data(
    simulation_directory: Path,
    feature_directory: Path,
    config: ModelingConfig,
    *,
    include_held_out_sample: bool,
) -> TrainingData | ValidationData:
    labels_path, features_path, boundary = _validate_source_contract(
        simulation_directory, feature_directory, config
    )
    train = _PartitionBuilder()
    calibration_fit = _PartitionBuilder()
    policy_selection = _PartitionBuilder()
    campaign_sides: dict[str, set[str]] = {}
    campaign_counts: dict[str, int] = {}
    held_out_rows: list[list[int]] = []
    held_out_seen = 0
    held_out_expected = EXPECTED_HELD_OUT_ROWS
    sample_indices = _sample_indices(held_out_expected, config.parity_sample_rows)
    previous_key: tuple[datetime, str] | None = None
    seen_event_ids: set[str] = set()
    row_count = 0

    try:
        with features_path.open("rb") as feature_stream, labels_path.open("rb") as label_stream:
            while True:
                feature_raw = feature_stream.readline()
                label_raw = label_stream.readline()
                if not feature_raw and not label_raw:
                    break
                if not feature_raw or not label_raw:
                    raise ModelingDataError("modeling_source_early_eof")
                row_count += 1
                feature_value = parse_canonical_jsonl_row(feature_raw)
                label_value = parse_canonical_jsonl_row(label_raw)
                split, occurred_text = _validate_join_metadata(feature_value, label_value)
                occurred_at = _parse_timestamp(occurred_text)
                event_id = str(feature_value["event_id"])
                if event_id in seen_event_ids:
                    raise ModelingDataError("modeling_duplicate_event_id")
                seen_event_ids.add(event_id)
                ordering_key = (occurred_at, event_id)
                if previous_key is not None and ordering_key <= previous_key:
                    raise ModelingDataError("modeling_source_order_invalid")
                previous_key = ordering_key

                if split == SplitName.TEST.value:
                    if include_held_out_sample and held_out_seen in sample_indices:
                        try:
                            record = FeatureRecord.model_validate(feature_value, strict=False)
                        except ValidationError:
                            raise ModelingDataError("modeling_held_out_feature_invalid") from None
                        values = record.features.model_dump()
                        held_out_rows.append([values[name] for name in FEATURE_NAMES])
                    held_out_seen += 1
                    continue

                feature, label = _parse_training_row(feature_value, label_value)
                if split == SplitName.TRAIN.value:
                    train.append(feature, label)
                    continue
                if split != SplitName.CALIBRATION.value:
                    raise ModelingDataError("modeling_partition_order_invalid")
                side = "calibration_fit" if feature.occurred_at < boundary else "policy_selection"
                if side == "calibration_fit":
                    calibration_fit.append(feature, label)
                else:
                    policy_selection.append(feature, label)
                if label.is_attack:
                    if label.campaign_id is None:
                        raise ModelingDataError("modeling_attack_campaign_missing")
                    campaign_counts[label.campaign_id] = (
                        campaign_counts.get(label.campaign_id, 0) + 1
                    )
                    campaign_sides.setdefault(label.campaign_id, set()).add(side)
    except OSError:
        raise ModelingDataError("modeling_source_read_failed") from None
    except ModelingArtifactError as error:
        raise ModelingDataError(str(error)) from None

    if row_count != EXPECTED_TOTAL_ROWS or held_out_seen != held_out_expected:
        raise ModelingDataError("modeling_source_row_count_invalid")
    if file_metadata(labels_path)["sha256"] != config.source_contract.labels_sha256:
        raise ModelingDataError("modeling_labels_changed_during_load")
    if file_metadata(features_path)["sha256"] != config.source_contract.features_sha256:
        raise ModelingDataError("modeling_features_changed_during_load")
    _validate_campaign_partition(campaign_sides, campaign_counts)

    result = TrainingData(
        train=train.finish(),
        calibration_fit=calibration_fit.finish(),
        policy_selection=policy_selection.finish(),
        boundary_timestamp=boundary.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{boundary.microsecond // 1_000:03d}Z",
    )
    if result.train.row_count != EXPECTED_TRAIN_ROWS:
        raise ModelingDataError("modeling_train_row_count_invalid")
    if (
        result.calibration_fit.row_count != EXPECTED_CALIBRATION_FIT_ROWS
        or result.policy_selection.row_count != EXPECTED_POLICY_SELECTION_ROWS
    ):
        raise ModelingDataError("modeling_calibration_subdivision_invalid")
    for partition in (result.train, result.calibration_fit, result.policy_selection):
        if set(partition.targets.tolist()) != {0, 1}:
            raise ModelingDataError("modeling_fit_partition_requires_both_classes")
    if include_held_out_sample:
        sample = np.asarray(held_out_rows, dtype=np.float64)
        if sample.shape != (len(sample_indices), FEATURE_COUNT):
            raise ModelingDataError("modeling_held_out_feature_sample_invalid")
        return ValidationData(training=result, held_out_feature_sample=sample)
    return result


def load_evaluation_data(
    simulation_directory: Path, feature_directory: Path, config: ModelingConfig
) -> EvaluationData:
    labels_path, features_path, _ = _validate_source_contract(
        simulation_directory, feature_directory, config
    )
    features: list[list[int]] = []
    targets: list[int] = []
    scenarios: list[str] = []
    campaigns: list[str | None] = []
    occurred_at_ms: list[int] = []
    seen_event_ids: set[str] = set()
    previous_key: tuple[datetime, str] | None = None
    row_count = 0
    try:
        with features_path.open("rb") as feature_stream, labels_path.open("rb") as label_stream:
            while True:
                feature_raw = feature_stream.readline()
                label_raw = label_stream.readline()
                if not feature_raw and not label_raw:
                    break
                if not feature_raw or not label_raw:
                    raise ModelingDataError("modeling_source_early_eof")
                row_count += 1
                feature_value = parse_canonical_jsonl_row(feature_raw)
                label_value = parse_canonical_jsonl_row(label_raw)
                split, occurred_text = _validate_join_metadata(feature_value, label_value)
                event_id = str(feature_value["event_id"])
                if event_id in seen_event_ids:
                    raise ModelingDataError("modeling_duplicate_event_id")
                seen_event_ids.add(event_id)
                occurred_at = _parse_timestamp(occurred_text)
                ordering_key = (occurred_at, event_id)
                if previous_key is not None and ordering_key <= previous_key:
                    raise ModelingDataError("modeling_source_order_invalid")
                previous_key = ordering_key
                if split != SplitName.TEST.value:
                    continue
                feature, label = _parse_training_row(feature_value, label_value)
                values = feature.features.model_dump()
                features.append([values[name] for name in FEATURE_NAMES])
                targets.append(int(label.is_attack))
                scenarios.append(label.scenario_type.value)
                campaigns.append(label.campaign_id)
                occurred_at_ms.append(_timestamp_ms(feature.occurred_at))
    except OSError:
        raise ModelingDataError("modeling_evaluation_source_read_failed") from None
    except ModelingArtifactError as error:
        raise ModelingDataError(str(error)) from None
    if file_metadata(labels_path)["sha256"] != config.source_contract.labels_sha256:
        raise ModelingDataError("modeling_labels_changed_during_evaluation")
    if file_metadata(features_path)["sha256"] != config.source_contract.features_sha256:
        raise ModelingDataError("modeling_features_changed_during_evaluation")
    if len(targets) != EXPECTED_HELD_OUT_ROWS:
        raise ModelingDataError("modeling_evaluation_held_out_row_count_invalid")
    if row_count != EXPECTED_TOTAL_ROWS or set(targets) != {0, 1}:
        raise ModelingDataError("modeling_evaluation_class_count_invalid")
    return EvaluationData(
        features=np.asarray(features, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.int8),
        scenarios=tuple(scenarios),
        campaign_ids=tuple(campaigns),
        occurred_at_ms=np.asarray(occurred_at_ms, dtype=np.int64),
    )
