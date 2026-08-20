import json
from pathlib import Path

import pytest

from riskloom.features import extraction as extraction_module
from riskloom.features.artifacts import FeatureArtifactError, FileMetadata
from riskloom.features.config import FeatureConfig
from riskloom.features.extraction import FeatureExtractionError, extract_feature_dataset
from riskloom.features.validation import validate_feature_dataset
from riskloom.simulation.artifacts import generate_dataset
from riskloom.simulation.config import load_generator_config
from riskloom.simulation.event_schema import CheckoutAttemptEvent


def test_extraction_streams_exact_rows_without_leaking_event_fields(
    tmp_path: Path,
    tiny_events_path: Path,
    tiny_events: list[CheckoutAttemptEvent],
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / "features"
    result = extract_feature_dataset(tiny_events_path, feature_config, output)
    assert result.row_count == len(tiny_events)
    assert result.feature_count == 75
    assert validate_feature_dataset(tiny_events_path, feature_config, output)["status"] == "valid"

    lines = (output / "features.jsonl").read_bytes().splitlines()
    assert len(lines) == len(tiny_events)
    first = json.loads(lines[0])
    assert set(first) == {"event_id", "features", "occurred_at"}
    assert len(first["features"]) == 75
    prohibited = {
        "merchant_id",
        "checkout_id",
        "customer_token",
        "device_token",
        "network_token",
        "session_token",
        "payment_instrument_token",
        "currency",
        "outcome",
        "failure_category",
        "label",
        "split",
        "scenario",
        "campaign",
        "risk_score",
        "prediction",
    }
    assert prohibited.isdisjoint(first["features"])
    all_bytes = b"".join(path.read_bytes() for path in output.iterdir())
    assert tiny_events[0].merchant_id.encode() not in all_bytes
    assert tiny_events[0].payment_instrument_token.encode() not in all_bytes
    assert b'"outcome"' not in all_bytes
    assert b'"failure_category"' not in all_bytes


def test_empty_and_invalid_source_fail_without_publication(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    for name, content in (("empty", b""), ("invalid", b"{invalid\n")):
        source = tmp_path / f"{name}.jsonl"
        source.write_bytes(content)
        output = tmp_path / f"{name}-output"
        with pytest.raises(FeatureExtractionError):
            extract_feature_dataset(source, feature_config, output)
        assert not output.exists()
        assert not any(
            path.name.startswith(".riskloom-features-staging-") for path in tmp_path.iterdir()
        )


def test_write_failure_aborts_and_discards_staging(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_serialization(value: object) -> bytes:
        del value
        raise OSError("synthetic-sensitive-write-marker")

    monkeypatch.setattr(extraction_module, "canonical_json_bytes", fail_serialization)
    output = tmp_path / "write-failure"
    with pytest.raises(FeatureExtractionError, match="failed") as exc_info:
        extract_feature_dataset(tiny_events_path, feature_config, output)
    assert "synthetic-sensitive-write-marker" not in str(exc_info.value)
    assert not output.exists()
    assert not any(
        path.name.startswith(".riskloom-features-staging-") for path in tmp_path.iterdir()
    )


def test_write_failure_retry_uses_a_fresh_engine_and_is_canonical(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = extraction_module.canonical_json_bytes
    calls = 0

    def fail_second_record(value: object) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-record write failure")
        return canonical(value)

    monkeypatch.setattr(extraction_module, "canonical_json_bytes", fail_second_record)
    output = tmp_path / "retried"
    with pytest.raises(FeatureExtractionError, match="write_failed"):
        extract_feature_dataset(tiny_events_path, feature_config, output)
    assert not output.exists()

    monkeypatch.undo()
    retried = extract_feature_dataset(tiny_events_path, feature_config, output)
    reference_output = tmp_path / "reference"
    reference = extract_feature_dataset(tiny_events_path, feature_config, reference_output)
    assert retried.feature_dataset_id == reference.feature_dataset_id
    for filename in ("features.jsonl", "report.json", "manifest.json"):
        assert (output / filename).read_bytes() == (reference_output / filename).read_bytes()


@pytest.mark.parametrize(
    ("changed_call", "error"),
    [(2, "changed_during_extraction"), (3, "changed_before_publication")],
)
def test_source_mutation_is_detected_before_publication(
    changed_call: int,
    error: str,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_metadata = extraction_module.file_metadata
    calls = 0

    def changing_metadata(path: Path) -> FileMetadata:
        nonlocal calls
        calls += 1
        actual = real_metadata(path)
        if calls == changed_call:
            return FileMetadata(actual.byte_size, "0" * 64)
        return actual

    monkeypatch.setattr(extraction_module, "file_metadata", changing_metadata)
    output = tmp_path / "changed-source"
    with pytest.raises(FeatureExtractionError, match=error):
        extract_feature_dataset(tiny_events_path, feature_config, output)
    assert not output.exists()


def test_source_mutation_during_existing_output_validation_blocks_overwrite(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing"
    extract_feature_dataset(tiny_events_path, feature_config, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    real_metadata = extraction_module.file_metadata
    calls = 0

    def changing_metadata(path: Path) -> FileMetadata:
        nonlocal calls
        calls += 1
        actual = real_metadata(path)
        if calls == 2:
            return FileMetadata(actual.byte_size, "f" * 64)
        return actual

    monkeypatch.setattr(extraction_module, "file_metadata", changing_metadata)
    with pytest.raises(FeatureExtractionError, match="existing_validation"):
        extract_feature_dataset(tiny_events_path, feature_config, output, overwrite=True)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_day2_smoke_extracts_exactly_two_thousand_rows(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    repository = Path(__file__).parents[3]
    simulation_config = load_generator_config(repository / "configs/simulation/smoke.json")
    simulation_output = tmp_path / "simulation-smoke"
    generate_dataset(simulation_config, 20260820, simulation_output)

    feature_output = tmp_path / "feature-smoke"
    result = extract_feature_dataset(
        simulation_output / "events.jsonl", feature_config, feature_output
    )
    assert result.row_count == 2_000
    with (feature_output / "features.jsonl").open("rb") as stream:
        assert sum(1 for _ in stream) == 2_000


def test_unreadable_source_is_safe(tmp_path: Path, feature_config: FeatureConfig) -> None:
    with pytest.raises(FeatureArtifactError, match="unreadable"):
        extract_feature_dataset(tmp_path / "missing.jsonl", feature_config, tmp_path / "out")
