import json
import random
import shutil
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from riskloom.simulation.artifacts import (
    ARTIFACT_FILENAMES,
    canonical_sha256,
    generate_dataset,
    sha256_file,
    write_canonical_json,
)
from riskloom.simulation.config import (
    GeneratorConfig,
    boundary_timestamp,
    configuration_fingerprint,
    effective_configuration,
    load_generator_config,
    validate_profile_contract,
)
from riskloom.simulation.generation import (
    GeneratedRecord,
    SimulationGenerator,
    generate_records,
)
from riskloom.simulation.label_schema import SplitName
from riskloom.simulation.validation import (
    DatasetValidationError,
    validate_dataset_directory,
    validate_records,
)


def _placement_config(tiny_config: GeneratorConfig) -> GeneratorConfig:
    data = tiny_config.model_dump(mode="json")
    data["config_schema_version"] = "1.1.0"
    for split in data["splits"]:
        split["duration_days"] = 2
        split["event_count"] = 1_000
        split["campaign_count"] = 10
    data["splits"][1]["campaign_placement"] = {
        "protected_boundary_basis_points": 6_000,
        "minimum_campaigns_before_boundary": 5,
        "minimum_campaigns_after_boundary": 5,
        "minimum_gap_seconds": 300,
        "maximum_sampling_attempts_per_campaign": 4_096,
    }
    return GeneratorConfig.model_validate(data)


def _development_config_data() -> dict[str, object]:
    repository_root = Path(__file__).parents[3]
    return json.loads(
        (repository_root / "configs/simulation/development.json").read_text(encoding="utf-8")
    )


def _calibration_bounds(config: GeneratorConfig) -> tuple[datetime, datetime, datetime]:
    calibration_start = config.start_at + timedelta(days=config.splits[0].duration_days)
    calibration_end = calibration_start + timedelta(days=config.splits[1].duration_days)
    placement = config.splits[1].campaign_placement
    assert placement is not None
    boundary = boundary_timestamp(
        calibration_start,
        calibration_end,
        placement.protected_boundary_basis_points,
    )
    return calibration_start, calibration_end, boundary


def _calibration_campaigns(
    records: list[GeneratedRecord],
) -> dict[str, list[GeneratedRecord]]:
    campaigns: dict[str, list[GeneratedRecord]] = defaultdict(list)
    for record in records:
        if record.label.split is SplitName.CALIBRATION and record.label.is_attack:
            assert record.label.campaign_id is not None
            campaigns[record.label.campaign_id].append(record)
    return campaigns


def _milliseconds(value: timedelta) -> int:
    return value.days * 86_400_000 + value.seconds * 1_000 + value.microseconds // 1_000


def test_shared_boundary_uses_floor_integer_milliseconds() -> None:
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    end = start + timedelta(milliseconds=10_001)

    assert boundary_timestamp(start, end, 6_000) == start + timedelta(milliseconds=6_000)


def test_configuration_1_0_remains_loadable_and_placement_requires_1_1(
    tiny_config: GeneratorConfig,
) -> None:
    assert tiny_config.config_schema_version == "1.0.0"
    assert all(split.campaign_placement is None for split in tiny_config.splits)

    invalid = tiny_config.model_dump(mode="json")
    invalid["splits"][1]["campaign_placement"] = {
        "protected_boundary_basis_points": 6_000,
        "minimum_campaigns_before_boundary": 2,
        "minimum_campaigns_after_boundary": 2,
        "minimum_gap_seconds": 300,
        "maximum_sampling_attempts_per_campaign": 4_096,
    }
    invalid["splits"][1]["campaign_count"] = 4
    with pytest.raises(ValueError, match="schema 1.1.0"):
        GeneratorConfig.model_validate(invalid)


def test_configuration_1_0_matches_frozen_head_artifact_hashes(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    expected = {
        "events.jsonl": (
            178_000,
            "d3e3c9ec7bd0bbd844baba00a1d26faeea7f5c50a03b5e773ca2c1fa3e9e90b7",
        ),
        "labels.jsonl": (
            74_239,
            "c5045446d96669ed13e92f0822e6a03a50e8c3753e98ba9dbad18aaecec86b2c",
        ),
        "manifest.json": (
            2_548,
            "5090b3e3c61d0fa9f2f8464e92ae8be60603b094f324c84f1344204cc2d7d0c4",
        ),
        "report.json": (4_344, "0b96cb255d5bbdffa0f505c48ccc59c5be362798182f328d3beeac2aea211658"),
    }
    output = tmp_path / "legacy-golden"

    result = generate_dataset(tiny_config, 1_234, output)

    assert result.dataset_id == "f1b902c7b7c32a1d4ebf9cc6a75d98289f4a41482d2a3b97649793b595c862a7"
    for filename in ARTIFACT_FILENAMES:
        expected_size, expected_hash = expected[filename]
        assert (output / filename).stat().st_size == expected_size
        assert sha256_file(output / filename) == expected_hash
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_version"] == "1.0.0"
    assert "effective_configuration_sha256" not in manifest
    assert all(
        "campaign_placement" not in split for split in manifest["effective_configuration"]["splits"]
    )


def test_development_profile_locks_ten_equal_calibration_campaigns() -> None:
    repository_root = Path(__file__).parents[3]
    config = load_generator_config(repository_root / "configs/simulation/development.json")
    calibration = config.splits[1]
    placement = calibration.campaign_placement

    assert config.config_schema_version == "1.1.0"
    assert calibration.campaign_count == 10
    assert config.scenario_counts(calibration)["attack"] == 340
    assert config.scenario_counts(calibration)["attack"] // calibration.campaign_count == 34
    assert placement is not None
    assert placement.protected_boundary_basis_points == 6_000
    assert placement.minimum_campaigns_before_boundary == 5
    assert placement.minimum_campaigns_after_boundary == 5
    assert placement.minimum_gap_seconds == 300


@pytest.mark.parametrize(
    "mutation",
    [
        "profile_only",
        "profile_and_boundary",
        "profile_and_campaign_count",
        "profile_and_development_scale_counts",
    ],
)
def test_development_contract_cannot_be_bypassed_by_smoke_relabeling(mutation: str) -> None:
    data = _development_config_data()
    data["dataset_profile"] = "smoke"
    splits = data["splits"]
    assert isinstance(splits, list)
    calibration = splits[1]
    assert isinstance(calibration, dict)
    placement = calibration["campaign_placement"]
    assert isinstance(placement, dict)
    if mutation == "profile_and_boundary":
        placement["protected_boundary_basis_points"] = 5_999
    elif mutation == "profile_and_campaign_count":
        calibration["campaign_count"] = 12
        placement["minimum_campaigns_before_boundary"] = 6
        placement["minimum_campaigns_after_boundary"] = 6
    elif mutation == "profile_and_development_scale_counts":
        for split, count in zip(splits, (65_000, 17_500, 17_500), strict=True):
            assert isinstance(split, dict)
            split["event_count"] = count

    with pytest.raises(ValueError, match="smoke_contract_total_event_count"):
        GeneratorConfig.model_validate(data)


def test_reduced_1_1_smoke_contract_is_path_independent(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    validate_profile_contract(config)
    misleading_path = tmp_path / "development.json"
    write_canonical_json(misleading_path, config.model_dump(mode="json"))

    loaded = load_generator_config(misleading_path)

    assert loaded == config
    assert loaded.total_events == 3_000
    assert [split.event_count for split in loaded.splits] == [1_000, 1_000, 1_000]
    assert [loaded.scenario_counts(split)["attack"] for split in loaded.splits] == [20, 20, 20]


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_downgrade_with_placement",
        "unknown_schema_version",
        "unknown_dataset_profile",
        "negative_event_count",
        "boolean_event_count",
        "fractional_event_count",
        "malformed_event_count",
        "negative_attack_count",
        "invalid_campaign_count",
        "invalid_placement_side_count",
        "invalid_duration",
        "invalid_attempt_limit",
        "development_relabelled_as_smoke",
        "unknown_nested_field",
        "incompatible_nested_model",
    ],
)
def test_generation_preflight_reconstructs_and_rejects_mutated_models(
    tiny_config: GeneratorConfig,
    mutation: str,
) -> None:
    config = _placement_config(tiny_config)
    placement = config.splits[1].campaign_placement
    assert placement is not None
    if mutation == "schema_downgrade_with_placement":
        config.config_schema_version = "1.0.0"
    elif mutation == "unknown_schema_version":
        config.config_schema_version = cast(Any, "9.9.9")
    elif mutation == "unknown_dataset_profile":
        config.dataset_profile = cast(Any, "unknown")
    elif mutation == "negative_event_count":
        config.splits[0].event_count = -100
    elif mutation == "boolean_event_count":
        config.splits[0].event_count = cast(Any, True)
    elif mutation == "fractional_event_count":
        config.splits[0].event_count = cast(Any, 100.5)
    elif mutation == "malformed_event_count":
        config.splits[0].event_count = cast(Any, "100")
    elif mutation == "negative_attack_count":
        config.scenario_weights.attack = -1
    elif mutation == "invalid_campaign_count":
        config.splits[0].campaign_count = 0
    elif mutation == "invalid_placement_side_count":
        config.splits[1].campaign_placement = placement.model_copy(
            update={"minimum_campaigns_before_boundary": 0}
        )
    elif mutation == "invalid_duration":
        config.splits[0].duration_days = 0
    elif mutation == "invalid_attempt_limit":
        config.splits[1].campaign_placement = placement.model_copy(
            update={"maximum_sampling_attempts_per_campaign": 0}
        )
    elif mutation == "development_relabelled_as_smoke":
        config = GeneratorConfig.model_validate(_development_config_data())
        config.dataset_profile = "smoke"
    elif mutation == "unknown_nested_field":
        replacement = config.splits[0].model_dump(mode="python")
        replacement["unexpected"] = "synthetic-invalid-field"
        config.splits[0] = cast(Any, replacement)
    else:
        config.splits[0] = config.splits[1].model_copy(deep=True)

    with pytest.raises(ValidationError):
        generate_records(config, 20_260_820)


def test_generation_snapshot_preserves_valid_caller_object(
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    before = config.model_dump(mode="python")

    records = generate_records(config, 20_260_820)

    assert config.model_dump(mode="python") == before
    validate_records(records, config)


def test_invalid_config_fails_before_output_or_staging_changes(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    config.splits[0].event_count = -100
    output = tmp_path / "unrelated-existing-directory"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("synthetic unrelated marker\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValidationError):
        generate_dataset(config, 20_260_820, output)

    assert list(output.iterdir()) == [marker]
    assert marker.read_text(encoding="utf-8") == "synthetic unrelated marker\n"
    assert not list(tmp_path.glob(".riskloom-simulation-staging-*"))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("total_events", "smoke_contract_total_event_count"),
        ("split_events", "smoke_contract_split_event_count"),
        ("total_duration", "smoke_contract_total_duration"),
        ("split_duration", "smoke_contract_split_duration"),
        ("campaign_count", "smoke_contract_campaign_count"),
        ("campaigns_before", "smoke_contract_campaigns_before_boundary"),
        ("campaigns_after", "smoke_contract_campaigns_after_boundary"),
        ("sampling_attempts", "smoke_contract_sampling_attempts"),
    ],
)
def test_smoke_contract_rejects_each_scale_or_placement_limit(
    tiny_config: GeneratorConfig,
    mutation: str,
    error: str,
) -> None:
    data = _placement_config(tiny_config).model_dump(mode="json")
    splits = data["splits"]
    placement = splits[1]["campaign_placement"]
    if mutation == "total_events":
        for split in splits:
            split["event_count"] = 1_100
    elif mutation == "split_events":
        splits[0]["event_count"] = 1_500
        splits[1]["event_count"] = 700
        splits[2]["event_count"] = 700
    elif mutation == "total_duration":
        splits[0]["duration_days"] = 3
        splits[1]["duration_days"] = 2
        splits[2]["duration_days"] = 2
    elif mutation == "split_duration":
        splits[0]["duration_days"] = 5
    elif mutation == "campaign_count":
        splits[0]["campaign_count"] = 11
    elif mutation == "campaigns_before":
        placement["minimum_campaigns_before_boundary"] = 6
        placement["minimum_campaigns_after_boundary"] = 4
    elif mutation == "campaigns_after":
        placement["minimum_campaigns_before_boundary"] = 4
        placement["minimum_campaigns_after_boundary"] = 6
    else:
        placement["maximum_sampling_attempts_per_campaign"] = 4_097

    with pytest.raises(ValueError, match=error):
        GeneratorConfig.model_validate(data)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("train_duration", "development_contract_train_duration"),
        ("train_events", "development_contract_train_event_count"),
        ("train_campaigns", "development_contract_train_campaign_count"),
        ("train_profile", "development_contract_train_campaign_profile"),
        ("calibration_duration", "development_contract_calibration_duration"),
        ("calibration_events", "development_contract_calibration_event_count"),
        ("calibration_campaigns", "development_contract_calibration_campaign_count"),
        ("calibration_profile", "development_contract_calibration_campaign_profile"),
        ("boundary", "development_contract_boundary"),
        ("before", "development_contract_campaigns_before"),
        ("after", "development_contract_campaigns_after"),
        ("gap", "development_contract_campaign_gap"),
        ("attempts", "development_contract_sampling_attempts"),
        ("test_duration", "development_contract_test_duration"),
        ("test_events", "development_contract_test_event_count"),
        ("test_campaigns", "development_contract_test_campaign_count"),
        ("test_profile", "development_contract_test_campaign_profile"),
        ("shift_multiplier", "development_contract_shift_multiplier"),
        ("shift_network_ratio", "development_contract_shift_network_ratio"),
        ("shift_network_presence", "development_contract_shift_network_presence"),
    ],
)
def test_development_contract_rejects_each_locked_field(case: str, error: str) -> None:
    repository_root = Path(__file__).parents[3]
    data = json.loads(
        (repository_root / "configs/simulation/development.json").read_text(encoding="utf-8")
    )
    train, calibration, test = data["splits"]
    placement = calibration["campaign_placement"]
    if case == "train_duration":
        train["duration_days"] = 19
    elif case == "train_events":
        train["event_count"] = 65_000
    elif case == "train_campaigns":
        train["campaign_count"] = 10
    elif case == "train_profile":
        train["campaign_profile"] = "entity_reuse_shift"
    elif case == "calibration_duration":
        calibration["duration_days"] = 6
    elif case == "calibration_events":
        calibration["event_count"] = 18_000
    elif case == "calibration_campaigns":
        calibration["campaign_count"] = 20
        placement["minimum_campaigns_before_boundary"] = 10
        placement["minimum_campaigns_after_boundary"] = 10
    elif case == "calibration_profile":
        calibration["campaign_profile"] = "entity_reuse_shift"
    elif case == "boundary":
        placement["protected_boundary_basis_points"] = 5_999
    elif case == "before":
        placement["minimum_campaigns_before_boundary"] = 4
    elif case == "after":
        placement["minimum_campaigns_after_boundary"] = 4
    elif case == "gap":
        placement["minimum_gap_seconds"] = 301
    elif case == "attempts":
        placement["maximum_sampling_attempts_per_campaign"] = 4_095
    elif case == "test_duration":
        test["duration_days"] = 6
    elif case == "test_events":
        test["event_count"] = 18_000
    elif case == "test_campaigns":
        test["campaign_count"] = 4
    elif case == "test_profile":
        test["campaign_profile"] = "baseline_reuse"
    elif case == "shift_multiplier":
        data["controlled_test_shift"]["minimum_unique_entity_ratio_multiplier"] = 3
    elif case == "shift_network_ratio":
        data["controlled_test_shift"]["maximum_unique_network_ratio_basis_points"] = 4_999
    else:
        data["controlled_test_shift"]["minimum_network_presence_basis_points"] = 9_001

    with pytest.raises(ValueError, match=error):
        GeneratorConfig.model_validate(data)


def test_seeded_irregular_placement_is_deterministic_and_boundary_safe(
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    first = generate_records(config, 20_260_820)
    second = generate_records(config, 20_260_820)
    validate_records(first, config)

    assert [record.event.model_dump(mode="json") for record in first] == [
        record.event.model_dump(mode="json") for record in second
    ]
    _, _, boundary = _calibration_bounds(config)
    campaigns = _calibration_campaigns(first)
    assert len(campaigns) == 10
    assert {len(group) for group in campaigns.values()} == {2}
    assert (
        sum(max(row.event.occurred_at for row in group) < boundary for group in campaigns.values())
        == 5
    )
    assert (
        sum(min(row.event.occurred_at for row in group) >= boundary for group in campaigns.values())
        == 5
    )

    envelopes = sorted(
        (
            min(record.event.occurred_at for record in group),
            max(record.event.occurred_at for record in group),
        )
        for group in campaigns.values()
    )
    gaps_ms = [
        _milliseconds(current[0] - previous[1])
        for previous, current in zip(envelopes, envelopes[1:], strict=False)
    ]
    assert min(gaps_ms) >= 300_000
    assert len(set(gaps_ms)) > 1


def test_different_seed_changes_valid_irregular_placement(
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    first = generate_records(config, 1)
    second = generate_records(config, 2)
    validate_records(first, config)
    validate_records(second, config)

    first_times = sorted(
        record.event.occurred_at
        for record in first
        if record.label.split is SplitName.CALIBRATION and record.label.is_attack
    )
    second_times = sorted(
        record.event.occurred_at
        for record in second
        if record.label.split is SplitName.CALIBRATION and record.label.is_attack
    )
    assert first_times != second_times


def test_configuration_fingerprint_binds_1_1_identifiers_streams_and_artifacts(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    changed_data = config.model_dump(mode="json")
    changed_data["currency"] = "USD"
    changed = GeneratorConfig.model_validate(changed_data)
    fingerprint = configuration_fingerprint(config)
    changed_fingerprint = configuration_fingerprint(changed)

    assert fingerprint == canonical_sha256(effective_configuration(config))
    assert changed_fingerprint == canonical_sha256(effective_configuration(changed))
    assert fingerprint != changed_fingerprint
    original_records = generate_records(config, 20_260_820)
    changed_records = generate_records(changed, 20_260_820)
    assert [record.event.event_id for record in original_records] != [
        record.event.event_id for record in changed_records
    ]
    assert [record.event.occurred_at for record in original_records] != [
        record.event.occurred_at for record in changed_records
    ]

    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(config, 20_260_820, first)
    generate_dataset(config, 20_260_820, second)
    for filename in ARTIFACT_FILENAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_version"] == "1.1.0"
    assert manifest["effective_configuration_sha256"] == fingerprint


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_fingerprint", "configuration_fingerprint_mismatch"),
        ("replaced_fingerprint", "configuration_fingerprint_mismatch"),
        ("malformed_fingerprint", "configuration_fingerprint_mismatch"),
        ("unsupported_generator", "generator_configuration_version_mismatch"),
        ("generator_config_mismatch", "generator_configuration_version_mismatch"),
        ("profile_contract_mismatch", "manifest_schema_invalid"),
    ],
)
def test_1_1_manifest_tampering_fails_closed_through_public_validation(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
    mutation: str,
    error: str,
) -> None:
    original = tmp_path / "original"
    generate_dataset(_placement_config(tiny_config), 20_260_820, original)
    validate_dataset_directory(original)
    candidate = tmp_path / mutation
    shutil.copytree(original, candidate)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing_fingerprint":
        del manifest["effective_configuration_sha256"]
    elif mutation == "replaced_fingerprint":
        manifest["effective_configuration_sha256"] = "0" * 64
    elif mutation == "malformed_fingerprint":
        manifest["effective_configuration_sha256"] = "not-a-sha256"
    elif mutation == "unsupported_generator":
        manifest["generator_version"] = "9.9.9"
    elif mutation == "generator_config_mismatch":
        manifest["config_schema_version"] = "1.0.0"
    else:
        manifest["effective_configuration"]["dataset_profile"] = "development"
    write_canonical_json(manifest_path, manifest)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("synthetic marker\n", encoding="utf-8", newline="\n")

    with pytest.raises(DatasetValidationError, match=error) as exc_info:
        validate_dataset_directory(candidate)

    assert exc_info.value.__cause__ is None
    assert unrelated.read_text(encoding="utf-8") == "synthetic marker\n"


def test_exact_boundary_belongs_to_policy_selection(
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    records = generate_records(config, 20_260_820)
    _, _, boundary = _calibration_bounds(config)
    campaigns = _calibration_campaigns(records)
    campaign_ids = sorted(campaigns)
    starts = (
        boundary - timedelta(hours=12),
        boundary - timedelta(hours=9),
        boundary - timedelta(hours=6, minutes=30),
        boundary - timedelta(hours=4),
        boundary - timedelta(hours=1, minutes=30),
        boundary,
        boundary + timedelta(hours=2),
        boundary + timedelta(hours=4, minutes=30),
        boundary + timedelta(hours=7, minutes=30),
        boundary + timedelta(hours=11),
    )
    replacements: dict[str, datetime] = {}
    for campaign_id, campaign_start in zip(campaign_ids, starts, strict=True):
        group = sorted(campaigns[campaign_id], key=lambda row: row.event.event_id)
        for index, record in enumerate(group):
            replacements[record.event.event_id] = campaign_start + timedelta(milliseconds=index)
    changed = [
        replace(
            record,
            event=record.event.model_copy(
                update={"occurred_at": replacements[record.event.event_id]}
            ),
        )
        if record.event.event_id in replacements
        else record
        for record in records
    ]
    changed.sort(key=lambda record: (record.event.occurred_at, record.event.event_id))

    validate_records(changed, config)
    exact_boundary_group = [
        record for record in changed if record.label.campaign_id == campaign_ids[5]
    ]
    assert min(record.event.occurred_at for record in exact_boundary_group) == boundary
    assert all(record.event.occurred_at >= boundary for record in exact_boundary_group)


def test_placement_exhaustion_cleans_only_owned_staging(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _placement_config(tiny_config).model_dump(mode="json")
    data["splits"][1]["campaign_placement"]["maximum_sampling_attempts_per_campaign"] = 1
    config = GeneratorConfig.model_validate(data)
    original_random_stream = SimulationGenerator._random_stream

    def collision_stream(self: SimulationGenerator, split: str, component: str) -> random.Random:
        if component.startswith("campaign-window-"):
            return random.Random(0)
        return original_random_stream(self, split, component)

    monkeypatch.setattr(SimulationGenerator, "_random_stream", collision_stream)
    unrelated = tmp_path / ".riskloom-simulation-staging-unrelated"
    unrelated.mkdir()
    marker = unrelated / "marker.txt"
    marker.write_text("synthetic unrelated marker", encoding="utf-8")
    output = tmp_path / "simulation"

    with pytest.raises(ValueError, match="campaign_placement_infeasible") as exc_info:
        generate_dataset(config, 20_260_820, output)

    assert exc_info.value.__cause__ is None
    assert not output.exists()
    assert marker.read_text(encoding="utf-8") == "synthetic unrelated marker"
    assert {
        path.name
        for path in tmp_path.iterdir()
        if path.name.startswith(".riskloom-simulation-staging-")
    } == {unrelated.name}


def test_validation_rejects_a_campaign_crossing_the_shared_boundary(
    tiny_config: GeneratorConfig,
) -> None:
    config = _placement_config(tiny_config)
    records = generate_records(config, 20_260_820)
    _, _, boundary = _calibration_bounds(config)
    campaigns = _calibration_campaigns(records)
    before_campaign = next(
        campaign_id
        for campaign_id, group in campaigns.items()
        if max(record.event.occurred_at for record in group) < boundary
    )
    crossing_record = max(
        campaigns[before_campaign],
        key=lambda record: record.event.occurred_at,
    )
    changed = [
        replace(
            record,
            event=record.event.model_copy(update={"occurred_at": boundary}),
        )
        if record.event.event_id == crossing_record.event.event_id
        else record
        for record in records
    ]
    changed.sort(key=lambda record: (record.event.occurred_at, record.event.event_id))

    with pytest.raises(DatasetValidationError, match="crosses_protected_boundary"):
        validate_records(changed, config)


def test_validation_rejects_campaign_gap_violation(tiny_config: GeneratorConfig) -> None:
    config = _placement_config(tiny_config)
    records = generate_records(config, 20_260_820)
    campaigns = _calibration_campaigns(records)
    ordered = sorted(
        (
            min(record.event.occurred_at for record in group),
            max(record.event.occurred_at for record in group),
            campaign_id,
        )
        for campaign_id, group in campaigns.items()
    )
    previous_end = ordered[0][1]
    target_id = ordered[1][2]
    target_start = ordered[1][0]
    shift = previous_end + timedelta(seconds=299) - target_start
    changed = [
        replace(
            record,
            event=record.event.model_copy(update={"occurred_at": record.event.occurred_at + shift}),
        )
        if record.label.campaign_id == target_id
        else record
        for record in records
    ]
    changed.sort(key=lambda record: (record.event.occurred_at, record.event.event_id))

    with pytest.raises(DatasetValidationError, match="campaign_placement_gap_invalid"):
        validate_records(changed, config)
