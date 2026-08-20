import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from riskloom.simulation.artifacts import write_canonical_json
from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.generation import GeneratedRecord, generate_records
from riskloom.simulation.identifiers import deterministic_identifier
from riskloom.simulation.label_schema import CampaignProfile, SplitName
from riskloom.simulation.validation import (
    DatasetValidationError,
    validate_dataset_directory,
    validate_records,
)


def test_configuration_rejects_rounded_scenario_counts(tiny_config: GeneratorConfig) -> None:
    data = tiny_config.model_dump(mode="json")
    data["splits"][0]["event_count"] = 150
    with pytest.raises(ValueError, match="multiple of 100"):
        GeneratorConfig.model_validate(data)


def test_configuration_requires_ordered_profiles(tiny_config: GeneratorConfig) -> None:
    data = tiny_config.model_dump(mode="json")
    data["splits"][0]["campaign_profile"] = "entity_reuse_shift"
    with pytest.raises(ValueError, match="isolation policy"):
        GeneratorConfig.model_validate(data)


def test_configuration_rejects_invalid_weights_and_campaign_policy(
    tiny_config: GeneratorConfig,
) -> None:
    negative = tiny_config.model_dump(mode="json")
    negative["scenario_weights"]["normal"] = -1
    negative["scenario_weights"]["attack"] += 7_001
    with pytest.raises(ValueError):
        GeneratorConfig.model_validate(negative)

    changed = tiny_config.model_dump(mode="json")
    changed["scenario_weights"]["normal"] -= 1
    changed["scenario_weights"]["attack"] += 1
    with pytest.raises(ValueError, match="approved distribution"):
        GeneratorConfig.model_validate(changed)

    undersized_campaigns = tiny_config.model_dump(mode="json")
    undersized_campaigns["splits"][0]["campaign_count"] = 2
    with pytest.raises(ValueError, match="at least two events"):
        GeneratorConfig.model_validate(undersized_campaigns)

    excessive_network_missingness = tiny_config.model_dump(mode="json")
    excessive_network_missingness["missingness_rates"]["network"] = 1_001
    with pytest.raises(ValueError, match="network missingness"):
        GeneratorConfig.model_validate(excessive_network_missingness)


def test_validation_rejects_report_or_hash_tampering(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["event_count"] = 1
    write_canonical_json(report_path, report)
    with pytest.raises(DatasetValidationError, match="report_recomputation|artifact_hash"):
        validate_dataset_directory(output)


def test_generated_features_are_not_perfectly_separable(
    tiny_records: list[GeneratedRecord],
) -> None:
    attacks = [record for record in tiny_records if record.label.is_attack]
    legitimate = [record for record in tiny_records if not record.label.is_attack]
    extractors = (
        lambda record: record.event.amount_subunits,
        lambda record: record.event.outcome,
        lambda record: record.event.failure_category,
        lambda record: record.event.channel,
        lambda record: record.event.customer_token is None,
        lambda record: record.event.device_token is None,
        lambda record: record.event.network_token is None,
    )
    for extractor in extractors:
        assert {extractor(row) for row in attacks}.issubset({extractor(row) for row in legitimate})


def test_seed_range_is_bounded(tiny_config: GeneratorConfig) -> None:
    with pytest.raises(ValueError, match="seed_out_of_range"):
        generate_records(tiny_config, -1)
    with pytest.raises(ValueError, match="seed_out_of_range"):
        generate_records(tiny_config, True)  # type: ignore[arg-type]


def test_non_object_manifest_fails_with_safe_validation_error(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    (output / "manifest.json").write_text("[]\n", encoding="utf-8", newline="\n")
    with pytest.raises(DatasetValidationError, match="manifest_schema_invalid"):
        validate_dataset_directory(output)


@pytest.mark.parametrize("field", ["device_token", "session_token"])
def test_validation_enforces_each_controlled_rotation_shift(
    field: str,
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    baseline = next(
        record
        for record in tiny_records
        if record.label.is_attack and record.label.split is SplitName.CALIBRATION
    )
    baseline_value = getattr(baseline.event, field)
    changed = [
        replace(
            record,
            event=record.event.model_copy(update={field: baseline_value}),
        )
        if record.label.is_attack and record.label.split is SplitName.TEST
        else record
        for record in tiny_records
    ]
    with pytest.raises(DatasetValidationError, match="rotation_shift_missing"):
        validate_records(changed, tiny_config)


@pytest.mark.parametrize("kind", ["sparse", "missing"])
def test_validation_preserves_attack_network_coordination(
    kind: str,
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    changed: list[GeneratedRecord] = []
    attack_index = 0
    for record in tiny_records:
        if not record.label.is_attack:
            changed.append(record)
            continue
        network = (
            deterministic_identifier("net", 1234, "adversarial-network", attack_index)
            if kind == "sparse"
            else None
        )
        changed.append(
            replace(record, event=record.event.model_copy(update={"network_token": network}))
        )
        attack_index += 1
    expected = "coordination_too_sparse" if kind == "sparse" else "presence_too_low"
    with pytest.raises(DatasetValidationError, match=expected):
        validate_records(changed, tiny_config)


def test_validation_rejects_campaign_identity_and_merchant_tampering(
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    target = next(record for record in tiny_records if record.label.is_attack)
    wrong_metadata = target.label.generator_metadata.model_copy(
        update={"campaign_profile": CampaignProfile.ENTITY_REUSE_SHIFT}
    )
    wrong_profile = [
        replace(
            record,
            label=record.label.model_copy(update={"generator_metadata": wrong_metadata}),
        )
        if record is target
        else record
        for record in tiny_records
    ]
    with pytest.raises(DatasetValidationError, match="campaign_profile_mismatch"):
        validate_records(wrong_profile, tiny_config)

    campaign_id = target.label.campaign_id
    campaign_rows = [record for record in tiny_records if record.label.campaign_id == campaign_id]
    merchant = campaign_rows[0].event.merchant_id
    one_merchant = [
        replace(record, event=record.event.model_copy(update={"merchant_id": merchant}))
        if record.label.campaign_id == campaign_id
        else record
        for record in tiny_records
    ]
    with pytest.raises(DatasetValidationError, match="merchant_span_invalid"):
        validate_records(one_merchant, tiny_config)


def test_validation_rejects_retry_and_visible_value_tampering(
    tiny_config: GeneratorConfig,
    tiny_records: list[GeneratedRecord],
) -> None:
    retries = [
        record for record in tiny_records if record.label.scenario_type.value == "legitimate_retry"
    ]
    first_instance = retries[0].label.generator_metadata.scenario_instance_id
    second_instance = next(
        record.label.generator_metadata.scenario_instance_id
        for record in retries
        if record.label.generator_metadata.scenario_instance_id != first_instance
    )
    first_checkout = retries[0].event.checkout_id
    reused_checkout = [
        replace(record, event=record.event.model_copy(update={"checkout_id": first_checkout}))
        if record.label.generator_metadata.scenario_instance_id == second_instance
        else record
        for record in tiny_records
    ]
    with pytest.raises(DatasetValidationError, match="retry_checkout_reused"):
        validate_records(reused_checkout, tiny_config)

    attack_only_amount = [
        replace(record, event=record.event.model_copy(update={"amount_subunits": 100_000_000}))
        if record.label.is_attack
        else record
        for record in tiny_records
    ]
    with pytest.raises(DatasetValidationError, match="single_feature"):
        validate_records(attack_only_amount, tiny_config)

    outside_split = list(tiny_records)
    first = outside_split[0]
    outside_split[0] = replace(
        first,
        event=first.event.model_copy(
            update={"occurred_at": tiny_config.start_at - timedelta(milliseconds=1)}
        ),
    )
    outside_split.sort(key=lambda record: (record.event.occurred_at, record.event.event_id))
    with pytest.raises(DatasetValidationError, match="outside_labeled_split"):
        validate_records(outside_split, tiny_config)


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("row_count", "artifact_row_count_mismatch"),
        ("metadata_extra", "manifest_artifact_metadata_invalid"),
        ("manifest_extra", "manifest_schema_invalid"),
        ("self_hash", "manifest_must_not_hash_itself"),
        ("marker", "manifest_marker_invalid"),
        ("artifacts_type", "manifest_artifact_set_invalid"),
        ("seed", "manifest_schema_invalid"),
        ("dataset_id", "dataset_identity_mismatch"),
        ("artifact_set", "manifest_artifact_set_invalid"),
        ("metadata_type", "manifest_artifact_metadata_invalid"),
        ("hash", "artifact_hash_mismatch"),
        ("size", "artifact_size_mismatch"),
        ("report_type", "report_schema_invalid"),
    ],
)
def test_manifest_metadata_is_strict(
    kind: str,
    error: str,
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if kind == "row_count":
        manifest["artifacts"]["events.jsonl"]["row_count"] = 1
    elif kind == "metadata_extra":
        manifest["artifacts"]["events.jsonl"]["unexpected"] = 1
    elif kind == "manifest_extra":
        manifest["unexpected"] = 1
    elif kind == "self_hash":
        manifest["artifacts"]["manifest.json"] = {}
    elif kind == "marker":
        manifest["product"] = "synthetic-other-product"
    elif kind == "artifacts_type":
        manifest["artifacts"] = []
    elif kind == "seed":
        manifest["seed"] = True
    elif kind == "dataset_id":
        manifest["dataset_id"] = "0" * 64
    elif kind == "artifact_set":
        del manifest["artifacts"]["labels.jsonl"]
    elif kind == "metadata_type":
        manifest["artifacts"]["events.jsonl"] = []
    elif kind == "hash":
        manifest["artifacts"]["events.jsonl"]["sha256"] = "0" * 64
    elif kind == "size":
        manifest["artifacts"]["events.jsonl"]["byte_size"] = 1
    else:
        write_canonical_json(output / "report.json", [])
    write_canonical_json(manifest_path, manifest)
    with pytest.raises(DatasetValidationError, match=error):
        validate_dataset_directory(output)
