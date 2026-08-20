import json
from collections.abc import Callable
from pathlib import Path

import pytest

from riskloom.features import validation as validation_module
from riskloom.features.artifacts import FileMetadata, canonical_json_bytes, file_metadata
from riskloom.features.config import FeatureConfig
from riskloom.features.extraction import FeatureExtractionError, extract_feature_dataset
from riskloom.features.validation import FeatureValidationError, validate_feature_dataset
from riskloom.simulation.event_schema import CheckoutAttemptEvent

EventFactory = Callable[..., CheckoutAttemptEvent]


@pytest.mark.parametrize("tamper", ["early_eof", "extra", "vector", "noncanonical"])
def test_streaming_lockstep_rejects_row_mismatches(
    tamper: str,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / tamper
    extract_feature_dataset(tiny_events_path, feature_config, output)
    feature_path = output / "features.jsonl"
    lines = feature_path.read_bytes().splitlines(keepends=True)
    if tamper == "early_eof":
        feature_path.write_bytes(b"".join(lines[:-1]))
        error = "early_eof"
    elif tamper == "extra":
        feature_path.write_bytes(b"".join([*lines, lines[-1]]))
        error = "extra"
    elif tamper == "vector":
        first = json.loads(lines[0])
        first["features"]["amount_subunits"] += 1
        lines[0] = canonical_json_bytes(first)
        feature_path.write_bytes(b"".join(lines))
        error = "vector_mismatch"
    else:
        first = json.loads(lines[0])
        lines[0] = (json.dumps(first, sort_keys=True) + "\n").encode()
        feature_path.write_bytes(b"".join(lines))
        error = "not_canonical"
    with pytest.raises(FeatureValidationError, match=error):
        validate_feature_dataset(tiny_events_path, feature_config, output)


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("malformed", "schema_invalid"),
        ("oversized", "line_oversized"),
        ("reordered", "vector_mismatch"),
    ],
)
def test_streaming_lockstep_rejects_malformed_oversized_and_reordered_rows(
    tamper: str,
    error: str,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / tamper
    extract_feature_dataset(tiny_events_path, feature_config, output)
    feature_path = output / "features.jsonl"
    lines = feature_path.read_bytes().splitlines(keepends=True)
    if tamper == "malformed":
        lines[0] = b"{malformed}\n"
    elif tamper == "oversized":
        lines[0] = b"{" + b"x" * 65_536 + b"\n"
    else:
        lines[0], lines[1] = lines[1], lines[0]
    feature_path.write_bytes(b"".join(lines))
    with pytest.raises(FeatureValidationError, match=error):
        validate_feature_dataset(tiny_events_path, feature_config, output)


@pytest.mark.parametrize("tamper", ["report", "manifest", "hash", "source"])
def test_report_and_manifest_are_fully_recomputed(
    tamper: str,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / tamper
    extract_feature_dataset(tiny_events_path, feature_config, output)
    if tamper == "report":
        path = output / "report.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["row_count"] += 1
    else:
        path = output / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "manifest":
            value["unexpected"] = 1
        elif tamper == "hash":
            value["artifacts"]["features.jsonl"]["sha256"] = "0" * 64
        else:
            value["source_events"]["sha256"] = "0" * 64
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(FeatureValidationError):
        validate_feature_dataset(tiny_events_path, feature_config, output)


@pytest.mark.parametrize("tamper", ["statistic", "diagnostic"])
def test_report_recomputation_detects_statistic_and_diagnostic_tampering(
    tamper: str,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / tamper
    extract_feature_dataset(tiny_events_path, feature_config, output)
    path = output / "report.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "statistic":
        value["features"]["amount_subunits"]["p50"] += 1
    else:
        value["state_diagnostics"]["indexes"]["merchant_60s"]["peak_entities"] += 1
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(FeatureValidationError, match="report_recomputation"):
        validate_feature_dataset(tiny_events_path, feature_config, output)


@pytest.mark.parametrize("filename", ["report.json", "manifest.json"])
def test_oversized_metadata_is_rejected_safely(
    filename: str,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / filename.removesuffix(".json")
    extract_feature_dataset(tiny_events_path, feature_config, output)
    (output / filename).write_bytes(b"{" + b"x" * 1_048_576 + b"\n")
    with pytest.raises(FeatureValidationError, match="metadata_oversized"):
        validate_feature_dataset(tiny_events_path, feature_config, output)


def test_source_change_during_streaming_validation_is_detected(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    extract_feature_dataset(tiny_events_path, feature_config, output)
    actual = file_metadata(tiny_events_path)
    calls = 0

    def changing_metadata(path: Path) -> FileMetadata:
        nonlocal calls
        del path
        calls += 1
        if calls == 2:
            return FileMetadata(actual.byte_size, "f" * 64)
        return actual

    monkeypatch.setattr(validation_module, "file_metadata", changing_metadata)
    with pytest.raises(FeatureValidationError, match="changed_during_validation"):
        validate_feature_dataset(tiny_events_path, feature_config, output)


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "blank", "oversized"])
def test_replay_boundary_rejects_invalid_source_streams(
    kind: str,
    tmp_path: Path,
    event_factory: EventFactory,
    write_events,
    feature_config: FeatureConfig,
) -> None:
    first = event_factory(1)
    second = event_factory(2, seconds=1)
    path = tmp_path / f"{kind}.jsonl"
    if kind == "duplicate":
        write_events(path, [first, first])
    elif kind == "unsorted":
        write_events(path, [second, first])
    elif kind == "blank":
        path.write_bytes(canonical_json_bytes(first.model_dump(mode="json")) + b"\n")
    else:
        path.write_bytes(b"{" + b"x" * 1_048_576 + b"\n")
    with pytest.raises(FeatureExtractionError):
        extract_feature_dataset(path, feature_config, tmp_path / f"{kind}-output")


def test_metadata_object_and_file_set_must_be_strict(
    tmp_path: Path, tiny_events_path: Path, feature_config: FeatureConfig
) -> None:
    output = tmp_path / "output"
    extract_feature_dataset(tiny_events_path, feature_config, output)
    (output / "unknown.json").write_bytes(b"{}\n")
    with pytest.raises(FeatureValidationError, match="artifact_set"):
        validate_feature_dataset(tiny_events_path, feature_config, output)

    (output / "unknown.json").unlink()
    (output / "manifest.json").write_bytes(b"[]\n")
    with pytest.raises(FeatureValidationError, match="metadata_schema"):
        validate_feature_dataset(tiny_events_path, feature_config, output)
