import ipaddress
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from riskloom.simulation.artifacts import (
    build_manifest,
    canonical_json_bytes,
    sha256_file,
    simulation_dataset_id,
)
from riskloom.simulation.config import (
    GeneratorConfig,
    SplitConfig,
    boundary_timestamp,
    configuration_fingerprint,
    generator_version_for_config,
    validate_profile_contract,
)
from riskloom.simulation.event_schema import CheckoutAttemptEvent, Outcome
from riskloom.simulation.generation import GeneratedRecord
from riskloom.simulation.label_schema import GroundTruthLabel, ScenarioType, SplitName
from riskloom.simulation.reporting import build_report

PROHIBITED_FIELDS = frozenset(
    {
        "address",
        "billing_address",
        "card_number",
        "contact",
        "cvc",
        "cvv",
        "email",
        "expiry",
        "ip_address",
        "pan",
        "phone",
        "raw_payload",
        "upi_id",
        "vpa",
    }
)
GROUND_TRUTH_FIELDS = frozenset(
    {"campaign_id", "generator_metadata", "is_attack", "scenario_type", "split"}
)
SAFE_IDENTIFIER_FIELDS = frozenset(
    {
        "checkout_id",
        "customer_token",
        "device_token",
        "event_id",
        "merchant_id",
        "network_token",
        "payment_instrument_token",
        "session_token",
    }
)
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HANDLE_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+$")
_CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEPARATOR_PATTERN = re.compile(r"[^a-z0-9]+")


class DatasetValidationError(ValueError):
    """Safe dataset validation error containing no artifact contents."""


def validate_prohibited_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            separated = _CAMEL_BOUNDARY_PATTERN.sub("_", str(key))
            normalized = _KEY_SEPARATOR_PATTERN.sub("_", separated.casefold()).strip("_")
            if normalized in PROHIBITED_FIELDS:
                raise DatasetValidationError(f"prohibited_field:{normalized}")
            validate_prohibited_keys(item)
    elif isinstance(value, list):
        for item in value:
            validate_prohibited_keys(item)


def validate_typed_string(field: str, value: str) -> None:
    if field in SAFE_IDENTIFIER_FIELDS:
        return
    if field == "occurred_at":
        return
    if _EMAIL_PATTERN.fullmatch(value) or _HANDLE_PATTERN.fullmatch(value):
        raise DatasetValidationError("prohibited_contact_shaped_value")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise DatasetValidationError("prohibited_network_address")
    if value.isdigit() and 13 <= len(value) <= 19:
        raise DatasetValidationError("prohibited_pan_shaped_value")
    if value.isdigit() and 3 <= len(value) <= 4:
        raise DatasetValidationError("prohibited_cvv_shaped_value")


def validate_event_privacy(event: CheckoutAttemptEvent) -> None:
    dumped = event.model_dump(mode="json")
    validate_prohibited_keys(dumped)
    if GROUND_TRUTH_FIELDS.intersection(dumped):
        raise DatasetValidationError("ground_truth_leak")
    for field, value in dumped.items():
        if isinstance(value, str):
            validate_typed_string(field, value)


def _split_boundaries(config: GeneratorConfig) -> dict[SplitName, tuple[datetime, datetime]]:
    boundaries: dict[SplitName, tuple[datetime, datetime]] = {}
    start = config.start_at.astimezone(UTC)
    for split in config.splits:
        end = start + timedelta(days=split.duration_days)
        boundaries[split.name] = (start, end)
        start = end
    return boundaries


def _timedelta_milliseconds(value: timedelta) -> int:
    return value.days * 86_400_000 + value.seconds * 1_000 + value.microseconds // 1_000


def _validate_campaign_placement(
    campaign_groups: Mapping[str, Sequence[GeneratedRecord]],
    split_start: datetime,
    split_end: datetime,
    split_config: SplitConfig,
    attack_count: int,
) -> None:
    placement = split_config.campaign_placement
    if placement is None:
        return
    if attack_count % split_config.campaign_count:
        raise DatasetValidationError("campaign_placement_quota_not_equal")
    expected_size = attack_count // split_config.campaign_count
    boundary = boundary_timestamp(
        split_start,
        split_end,
        placement.protected_boundary_basis_points,
    )
    before_count = 0
    after_count = 0
    envelopes: list[tuple[datetime, datetime]] = []
    for group in campaign_groups.values():
        timestamps = sorted(record.event.occurred_at for record in group)
        if len(timestamps) != expected_size:
            raise DatasetValidationError("campaign_placement_quota_not_equal")
        first = timestamps[0]
        last = timestamps[-1]
        if last < boundary:
            before_count += 1
        elif first >= boundary:
            after_count += 1
        else:
            raise DatasetValidationError("campaign_crosses_protected_boundary")
        envelopes.append((first, last))
    if before_count < placement.minimum_campaigns_before_boundary:
        raise DatasetValidationError("campaigns_before_boundary_missing")
    if after_count < placement.minimum_campaigns_after_boundary:
        raise DatasetValidationError("campaigns_after_boundary_missing")
    ordered = sorted(envelopes)
    gaps_ms = [
        _timedelta_milliseconds(current[0] - previous[1])
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ]
    minimum_gap_ms = placement.minimum_gap_seconds * 1_000
    if any(gap < minimum_gap_ms for gap in gaps_ms):
        raise DatasetValidationError("campaign_placement_gap_invalid")
    if len(gaps_ms) > 1 and len(set(gaps_ms)) == 1:
        raise DatasetValidationError("campaign_placement_not_irregular")


def validate_records(records: Sequence[GeneratedRecord], config: GeneratorConfig) -> None:
    if len(records) != config.total_events:
        raise DatasetValidationError("event_count_mismatch")
    event_ids = [record.event.event_id for record in records]
    if len(set(event_ids)) != len(event_ids):
        raise DatasetValidationError("duplicate_event_id")
    if event_ids != [record.label.event_id for record in records]:
        raise DatasetValidationError("event_label_join_mismatch")
    ordering = [(record.event.occurred_at, record.event.event_id) for record in records]
    if ordering != sorted(ordering):
        raise DatasetValidationError("events_not_chronological")

    boundaries = _split_boundaries(config)
    split_records: dict[SplitName, list[GeneratedRecord]] = defaultdict(list)
    for record in records:
        matching_splits = [
            split
            for split, (start, end) in boundaries.items()
            if start <= record.event.occurred_at < end
        ]
        if matching_splits != [record.label.split]:
            raise DatasetValidationError("event_outside_labeled_split")
        if record.event.outcome is Outcome.AUTHORIZED and record.event.failure_category is not None:
            raise DatasetValidationError("authorized_event_has_failure_category")
        if record.event.outcome is Outcome.FAILED and record.event.failure_category is None:
            raise DatasetValidationError("failed_event_missing_failure_category")
        validate_event_privacy(record.event)
        split_records[record.label.split].append(record)

    for split in config.splits:
        rows = split_records[split.name]
        if len(rows) != split.event_count:
            raise DatasetValidationError("split_count_mismatch")
        actual = Counter(record.label.scenario_type.value for record in rows)
        expected = config.scenario_counts(split)
        expected_names = {
            "attack": ScenarioType.CARD_TESTING_CAMPAIGN.value,
            "flash_sale": ScenarioType.FLASH_SALE.value,
            "legitimate_failure": ScenarioType.LEGITIMATE_FAILURE.value,
            "legitimate_retry": ScenarioType.LEGITIMATE_RETRY.value,
            "normal": ScenarioType.NORMAL.value,
            "shared_infrastructure": ScenarioType.SHARED_INFRASTRUCTURE.value,
        }
        translated = {expected_names[key]: value for key, value in expected.items()}
        if actual != Counter(translated):
            raise DatasetValidationError("scenario_quota_mismatch")

    retry_groups: dict[str, list[GeneratedRecord]] = defaultdict(list)
    for record in records:
        if record.label.scenario_type is ScenarioType.LEGITIMATE_RETRY:
            instance_id = record.label.generator_metadata.scenario_instance_id
            if instance_id is None:
                raise DatasetValidationError("retry_missing_instance")
            retry_groups[instance_id].append(record)
    retry_checkouts: set[str] = set()
    for group in retry_groups.values():
        ordered = sorted(group, key=lambda record: record.event.occurred_at)
        if not 2 <= len(ordered) <= 4:
            raise DatasetValidationError("retry_chain_size_invalid")
        first = ordered[0].event
        if first.outcome is not Outcome.FAILED:
            raise DatasetValidationError("retry_chain_must_start_failed")
        if first.checkout_id in retry_checkouts:
            raise DatasetValidationError("retry_checkout_reused_across_chains")
        retry_checkouts.add(first.checkout_id)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            gap = current.event.occurred_at - previous.event.occurred_at
            gap_milliseconds = (
                gap.days * 86_400_000 + gap.seconds * 1_000 + gap.microseconds // 1_000
            )
            if not (
                config.retry_bounds.minimum_gap_seconds * 1_000
                <= gap_milliseconds
                <= config.retry_bounds.maximum_gap_seconds * 1_000
            ):
                raise DatasetValidationError("retry_chain_gap_invalid")
        for field in (
            "checkout_id",
            "customer_token",
            "device_token",
            "network_token",
            "payment_instrument_token",
            "session_token",
        ):
            if len({getattr(record.event, field) for record in ordered}) != 1:
                raise DatasetValidationError("retry_chain_identity_changed")

    split_attacks = {
        split: [record for record in split_records[split] if record.label.is_attack]
        for split in SplitName
    }
    for split in config.splits:
        campaign_groups: dict[str, list[GeneratedRecord]] = defaultdict(list)
        for record in split_attacks[split.name]:
            campaign_id = record.label.campaign_id
            if campaign_id is None:
                raise DatasetValidationError("attack_campaign_id_missing")
            if record.label.generator_metadata.campaign_profile is not split.campaign_profile:
                raise DatasetValidationError("campaign_profile_mismatch")
            campaign_groups[campaign_id].append(record)
        if len(campaign_groups) != split.campaign_count:
            raise DatasetValidationError("campaign_count_mismatch")
        if sum(map(len, campaign_groups.values())) != config.scenario_counts(split)["attack"]:
            raise DatasetValidationError("campaign_event_quota_mismatch")
        for group in campaign_groups.values():
            if not group:
                raise DatasetValidationError("empty_campaign")
            if len(group) >= 2 and len({record.event.merchant_id for record in group}) < 2:
                raise DatasetValidationError("campaign_merchant_span_invalid")
            scenario_ids = {
                record.label.generator_metadata.scenario_instance_id for record in group
            }
            if None in scenario_ids or len(scenario_ids) != 1:
                raise DatasetValidationError("campaign_scenario_identity_mismatch")
        split_start, split_end = boundaries[split.name]
        _validate_campaign_placement(
            campaign_groups,
            split_start,
            split_end,
            split,
            config.scenario_counts(split)["attack"],
        )

    policy = config.controlled_test_shift
    test_attacks = split_attacks[SplitName.TEST]
    for field in ("device_token", "session_token"):
        test_unique = len(
            {
                getattr(record.event, field)
                for record in test_attacks
                if getattr(record.event, field) is not None
            }
        )
        if test_unique == 0:
            raise DatasetValidationError("test_shift_entity_missing")
        for baseline_split in (SplitName.TRAIN, SplitName.CALIBRATION):
            baseline = split_attacks[baseline_split]
            baseline_unique = len(
                {
                    getattr(record.event, field)
                    for record in baseline
                    if getattr(record.event, field) is not None
                }
            )
            if baseline_unique == 0 or (
                test_unique * len(baseline)
                < policy.minimum_unique_entity_ratio_multiplier
                * baseline_unique
                * len(test_attacks)
            ):
                raise DatasetValidationError("test_entity_rotation_shift_missing")

    for attacks in split_attacks.values():
        network_tokens = [
            record.event.network_token
            for record in attacks
            if record.event.network_token is not None
        ]
        unique_networks = len(set(network_tokens))
        if unique_networks * 10_000 > policy.maximum_unique_network_ratio_basis_points * len(
            attacks
        ):
            raise DatasetValidationError("attack_network_coordination_too_sparse")
        if len(network_tokens) * 10_000 < policy.minimum_network_presence_basis_points * len(
            attacks
        ):
            raise DatasetValidationError("attack_network_presence_too_low")

    attack_rows = [record for record in records if record.label.is_attack]
    legitimate_rows = [record for record in records if not record.label.is_attack]
    feature_extractors = (
        lambda record: record.event.amount_subunits,
        lambda record: record.event.outcome.value,
        lambda record: (
            record.event.failure_category.value
            if record.event.failure_category is not None
            else None
        ),
        lambda record: record.event.channel.value,
        lambda record: record.event.customer_token is None,
        lambda record: record.event.device_token is None,
        lambda record: record.event.network_token is None,
    )
    for extractor in feature_extractors:
        attack_values = {extractor(record) for record in attack_rows}
        legitimate_values = {extractor(record) for record in legitimate_rows}
        if not attack_values.issubset(legitimate_values):
            raise DatasetValidationError("single_feature_perfectly_separates_attack")


def read_records(directory: Path) -> tuple[list[GeneratedRecord], dict[str, Any], dict[str, Any]]:
    try:
        event_bytes = (directory / "events.jsonl").read_bytes()
        label_bytes = (directory / "labels.jsonl").read_bytes()
        report_bytes = (directory / "report.json").read_bytes()
        manifest_bytes = (directory / "manifest.json").read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DatasetValidationError("artifact_unreadable") from None
    for content in (event_bytes, label_bytes, report_bytes, manifest_bytes):
        if not content.endswith(b"\n") or b"\r" in content:
            raise DatasetValidationError("artifact_not_canonical")
    try:
        canonical_report = canonical_json_bytes(report)
        canonical_manifest = canonical_json_bytes(manifest)
    except (TypeError, ValueError):
        raise DatasetValidationError("artifact_not_canonical") from None
    if report_bytes != canonical_report or manifest_bytes != canonical_manifest:
        raise DatasetValidationError("artifact_not_canonical")
    event_lines = event_bytes.splitlines()
    label_lines = label_bytes.splitlines()
    if len(event_lines) != len(label_lines):
        raise DatasetValidationError("event_label_row_count_mismatch")
    records: list[GeneratedRecord] = []
    try:
        for event_line, label_line in zip(event_lines, label_lines, strict=True):
            event = CheckoutAttemptEvent.model_validate_json(event_line)
            label = GroundTruthLabel.model_validate_json(label_line)
            if event_line + b"\n" != canonical_json_bytes(event.model_dump(mode="json")):
                raise DatasetValidationError("artifact_not_canonical")
            if label_line + b"\n" != canonical_json_bytes(label.model_dump(mode="json")):
                raise DatasetValidationError("artifact_not_canonical")
            records.append(GeneratedRecord(event, label))
    except DatasetValidationError:
        raise
    except (ValidationError, ValueError):
        raise DatasetValidationError("artifact_schema_invalid") from None
    return records, report, manifest


def validate_dataset_directory(directory: Path) -> dict[str, Any]:
    records, report, manifest = read_records(directory)
    if not isinstance(manifest, dict):
        raise DatasetValidationError("manifest_schema_invalid")
    if manifest.get("product") != "RiskLoom" or manifest.get("artifact_type") != (
        "synthetic_checkout_simulation"
    ):
        raise DatasetValidationError("manifest_marker_invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DatasetValidationError("manifest_artifact_set_invalid")
    if "manifest.json" in artifacts:
        raise DatasetValidationError("manifest_must_not_hash_itself")
    try:
        seed_value = manifest["seed"]
        if type(seed_value) is not int or not 0 <= seed_value <= 2**63 - 1:
            raise ValueError("invalid seed")
        seed = seed_value
        config = GeneratorConfig.model_validate(manifest["effective_configuration"])
        validate_profile_contract(config)
        generator_version = manifest["generator_version"]
        if not isinstance(generator_version, str):
            raise ValueError("invalid generator version")
        declared_config_version = manifest["config_schema_version"]
        if not isinstance(declared_config_version, str):
            raise ValueError("invalid configuration version")
        dataset_id = manifest["dataset_id"]
        if not isinstance(dataset_id, str) or re.fullmatch(r"[0-9a-f]{64}", dataset_id) is None:
            raise ValueError("invalid dataset id")
    except (KeyError, TypeError, ValueError, ValidationError):
        raise DatasetValidationError("manifest_schema_invalid") from None
    if (
        declared_config_version != config.config_schema_version
        or generator_version != generator_version_for_config(config)
    ):
        raise DatasetValidationError("generator_configuration_version_mismatch")
    expected_fingerprint = configuration_fingerprint(config)
    if (
        expected_fingerprint is not None
        and manifest.get("effective_configuration_sha256") != expected_fingerprint
    ):
        raise DatasetValidationError("configuration_fingerprint_mismatch")
    expected_dataset_id = simulation_dataset_id(config, seed)
    if dataset_id != expected_dataset_id:
        raise DatasetValidationError("dataset_identity_mismatch")
    validate_records(records, config)
    if not isinstance(report, dict):
        raise DatasetValidationError("report_schema_invalid")
    if report != build_report(records, dataset_id, config):
        raise DatasetValidationError("report_recomputation_mismatch")
    expected_files = {"events.jsonl", "labels.jsonl", "report.json"}
    if set(artifacts) != expected_files:
        raise DatasetValidationError("manifest_artifact_set_invalid")
    actual_metadata: dict[str, dict[str, Any]] = {}
    expected_row_counts = {
        "events.jsonl": len(records),
        "labels.jsonl": len(records),
        "report.json": 1,
    }
    for filename in sorted(expected_files):
        metadata = artifacts[filename]
        path = directory / filename
        if not isinstance(metadata, dict):
            raise DatasetValidationError("manifest_artifact_metadata_invalid")
        try:
            actual_hash = sha256_file(path)
            actual_size = path.stat().st_size
        except OSError:
            raise DatasetValidationError("artifact_unreadable") from None
        if metadata.get("sha256") != actual_hash:
            raise DatasetValidationError("artifact_hash_mismatch")
        if metadata.get("byte_size") != actual_size:
            raise DatasetValidationError("artifact_size_mismatch")
        if metadata.get("row_count") != expected_row_counts[filename]:
            raise DatasetValidationError("artifact_row_count_mismatch")
        if set(metadata) != {"byte_size", "row_count", "sha256"}:
            raise DatasetValidationError("manifest_artifact_metadata_invalid")
        actual_metadata[filename] = {
            "byte_size": actual_size,
            "row_count": expected_row_counts[filename],
            "sha256": actual_hash,
        }
    if manifest != build_manifest(config, seed, dataset_id, actual_metadata):
        raise DatasetValidationError("manifest_schema_invalid")
    return {
        "dataset_id": dataset_id,
        "event_count": len(records),
        "seed": seed,
        "status": "valid",
    }
