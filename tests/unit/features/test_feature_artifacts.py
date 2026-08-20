import json
from collections.abc import Callable
from pathlib import Path

import pytest

from riskloom.features import artifacts as artifact_module
from riskloom.features.artifacts import (
    FEATURE_ARTIFACT_FILENAMES,
    FeatureArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    cleanup_staging_directory,
    file_metadata,
)
from riskloom.features.config import FeatureConfig
from riskloom.features.extraction import extract_feature_dataset
from riskloom.features.validation import FeatureValidationError, validate_feature_dataset
from riskloom.simulation.event_schema import CheckoutAttemptEvent

EventFactory = Callable[..., CheckoutAttemptEvent]


def test_same_input_produces_byte_identical_artifacts_and_manifest(
    tmp_path: Path, tiny_events_path: Path, feature_config: FeatureConfig
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = extract_feature_dataset(tiny_events_path, feature_config, first)
    second_result = extract_feature_dataset(tiny_events_path, feature_config, second)
    assert first_result.feature_dataset_id == second_result.feature_dataset_id
    for filename in FEATURE_ARTIFACT_FILENAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
        assert (first / filename).read_bytes().endswith(b"\n")
        assert b"\r" not in (first / filename).read_bytes()

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {"features.jsonl", "report.json"}
    assert "manifest.json" not in manifest["artifacts"]
    assert "output_directory" not in manifest
    assert "seed" not in manifest
    assert manifest["feature_count"] == 75
    assert manifest["source_events"] == file_metadata(tiny_events_path).as_dict()


def test_changed_source_changes_dataset_identity(
    tmp_path: Path,
    tiny_events: list[CheckoutAttemptEvent],
    write_events,
    event_factory: EventFactory,
    feature_config: FeatureConfig,
) -> None:
    first_source = write_events(tmp_path / "first.jsonl", tiny_events)
    second_source = write_events(
        tmp_path / "second.jsonl", [*tiny_events, event_factory(20, seconds=100)]
    )
    first = extract_feature_dataset(first_source, feature_config, tmp_path / "first-output")
    second = extract_feature_dataset(second_source, feature_config, tmp_path / "second-output")
    assert first.feature_dataset_id != second.feature_dataset_id


def test_guarded_overwrite_requires_same_source_and_config(
    tmp_path: Path,
    tiny_events: list[CheckoutAttemptEvent],
    tiny_events_path: Path,
    write_events,
    event_factory: EventFactory,
    feature_config: FeatureConfig,
) -> None:
    output = tmp_path / "output"
    first = extract_feature_dataset(tiny_events_path, feature_config, output)
    repeated = extract_feature_dataset(tiny_events_path, feature_config, output, overwrite=True)
    assert repeated.feature_dataset_id == first.feature_dataset_id

    changed_source = write_events(
        tmp_path / "changed.jsonl", [*tiny_events, event_factory(20, seconds=100)]
    )
    with pytest.raises(FeatureArtifactError, match="existing_feature_dataset_invalid"):
        extract_feature_dataset(changed_source, feature_config, output, overwrite=True)

    changed_config = feature_config.model_copy(update={"failure_rate_window_seconds": 301})
    with pytest.raises(FeatureArtifactError, match="existing_feature_dataset_invalid"):
        extract_feature_dataset(tiny_events_path, changed_config, output, overwrite=True)

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "unrelated.txt").write_text("synthetic marker", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="unknown_files"):
        extract_feature_dataset(tiny_events_path, feature_config, unknown, overwrite=True)
    assert (unknown / "unrelated.txt").is_file()


def test_partial_or_invalid_existing_output_is_never_replaced(
    tmp_path: Path, tiny_events_path: Path, feature_config: FeatureConfig
) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    (output / "features.jsonl").write_bytes(b"{}\n")
    with pytest.raises(FeatureArtifactError, match="unknown_files"):
        extract_feature_dataset(tiny_events_path, feature_config, output, overwrite=True)
    assert (output / "features.jsonl").read_bytes() == b"{}\n"

    valid = tmp_path / "invalid-known"
    extract_feature_dataset(tiny_events_path, feature_config, valid)
    (valid / "report.json").write_bytes(b"{}\n")
    with pytest.raises(FeatureArtifactError, match="existing_feature_dataset_invalid"):
        extract_feature_dataset(tiny_events_path, feature_config, valid, overwrite=True)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    marker = nonempty / "unrelated.txt"
    marker.write_text("synthetic marker", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="not_empty"):
        extract_feature_dataset(tiny_events_path, feature_config, nonempty)
    assert marker.read_text(encoding="utf-8") == "synthetic marker"


def test_unsafe_output_paths_and_symlink_ancestors_are_rejected(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FeatureArtifactError, match="unsafe_feature_output_directory"):
        extract_feature_dataset(tiny_events_path, feature_config, Path.cwd())
    with pytest.raises(FeatureArtifactError, match="unsafe_feature_output_directory"):
        extract_feature_dataset(tiny_events_path, feature_config, Path.home())
    with pytest.raises(FeatureArtifactError, match="unsafe_feature_output_directory"):
        extract_feature_dataset(tiny_events_path, feature_config, Path(tmp_path.anchor))

    file_output = tmp_path / "file"
    file_output.write_text("synthetic marker", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="not_directory"):
        extract_feature_dataset(tiny_events_path, feature_config, file_output)

    linked_parent = (tmp_path / "linked").absolute()
    original = Path.is_symlink

    def synthetic_symlink(path: Path) -> bool:
        return path.absolute() == linked_parent or original(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_symlink)
    with pytest.raises(FeatureArtifactError, match="symlinked"):
        extract_feature_dataset(tiny_events_path, feature_config, linked_parent / "out")


def test_manifest_is_published_last_and_partial_publication_is_untrusted(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[str] = []
    original_replace = artifact_module.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        published.append(Path(destination).name)
        original_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", recording_replace)
    output = tmp_path / "ordered"
    extract_feature_dataset(tiny_events_path, feature_config, output)
    assert published[-3:] == ["features.jsonl", "report.json", "manifest.json"]

    monkeypatch.undo()
    calls = 0

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic-publication-marker")
        original_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", failing_replace)
    partial = tmp_path / "publication-failure"
    with pytest.raises(FeatureArtifactError, match="publication_failed") as exc_info:
        extract_feature_dataset(tiny_events_path, feature_config, partial)
    assert "synthetic-publication-marker" not in str(exc_info.value)
    assert not (partial / "manifest.json").exists()
    with pytest.raises(FeatureValidationError, match="artifact_set"):
        validate_feature_dataset(tiny_events_path, feature_config, partial)


@pytest.mark.parametrize("failed_step", [1, 2, 3])
def test_each_publication_failure_leaves_no_complete_dataset(
    failed_step: int,
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = artifact_module.os.replace
    calls = 0

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_step:
            raise OSError("synthetic-publication-step-marker")
        original_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", failing_replace)
    output = tmp_path / f"failed-{failed_step}"
    neighbor = tmp_path / "unrelated.txt"
    neighbor.write_text("untouched", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="publication_failed"):
        extract_feature_dataset(tiny_events_path, feature_config, output)
    assert not (output / "manifest.json").exists()
    assert neighbor.read_text(encoding="utf-8") == "untouched"
    with pytest.raises(FeatureValidationError, match="artifact_set"):
        validate_feature_dataset(tiny_events_path, feature_config, output)


def test_overwrite_removes_only_manifest_immediately_before_known_replacements(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "overwrite-order"
    extract_feature_dataset(tiny_events_path, feature_config, output)
    actions: list[str] = []
    original_unlink = Path.unlink
    original_replace = artifact_module.os.replace

    def recording_unlink(path: Path, *args, **kwargs) -> None:
        actions.append(f"unlink:{path.name}")
        original_unlink(path, *args, **kwargs)

    def recording_replace(source: Path, destination: Path) -> None:
        actions.append(f"replace:{Path(destination).name}")
        original_replace(source, destination)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    monkeypatch.setattr(artifact_module.os, "replace", recording_replace)
    extract_feature_dataset(tiny_events_path, feature_config, output, overwrite=True)
    assert actions[-4:] == [
        "unlink:manifest.json",
        "replace:features.jsonl",
        "replace:report.json",
        "replace:manifest.json",
    ]


def test_failed_staged_validation_does_not_remove_old_manifest(
    tmp_path: Path,
    tiny_events_path: Path,
    feature_config: FeatureConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from riskloom.features import validation as validation_module

    output = tmp_path / "preserved"
    extract_feature_dataset(tiny_events_path, feature_config, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    original_validate = validation_module.validate_feature_dataset
    calls = 0

    def fail_staging_validation(
        events_path: Path, config: FeatureConfig, directory: Path
    ) -> dict[str, int | str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FeatureValidationError("synthetic_staged_validation_failure")
        return original_validate(events_path, config, directory)

    monkeypatch.setattr(validation_module, "validate_feature_dataset", fail_staging_validation)
    with pytest.raises(FeatureValidationError, match="staged_validation"):
        extract_feature_dataset(tiny_events_path, feature_config, output, overwrite=True)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_cleanup_refuses_any_target_other_than_owned_staging_directory(
    tmp_path: Path,
) -> None:
    requested_output = tmp_path / "requested-output"
    requested_output.mkdir()
    marker = requested_output / "unrelated.txt"
    marker.write_text("untouched", encoding="utf-8")
    with pytest.raises(FeatureArtifactError, match="unsafe_feature_staging_cleanup_target"):
        cleanup_staging_directory(requested_output, tmp_path)
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_canonical_helpers_reject_unserializable_values() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    with pytest.raises(FeatureArtifactError, match="canonicalizable"):
        canonical_json_bytes({"bad": object()})
