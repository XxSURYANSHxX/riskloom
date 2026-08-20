import json
from pathlib import Path

import pytest

import riskloom.simulation.artifacts as artifact_module
from riskloom.simulation.artifacts import (
    ARTIFACT_FILENAMES,
    ArtifactPublishError,
    generate_dataset,
    sha256_file,
    write_canonical_json,
    write_event_jsonl,
)
from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.validation import DatasetValidationError, validate_dataset_directory


def test_same_seed_produces_byte_identical_artifacts(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = generate_dataset(tiny_config, 9_999, first)
    second_result = generate_dataset(tiny_config, 9_999, second)
    assert first_result.dataset_id == second_result.dataset_id
    for filename in ARTIFACT_FILENAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_different_seed_changes_all_artifacts(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(tiny_config, 1, first)
    generate_dataset(tiny_config, 2, second)
    for filename in ARTIFACT_FILENAMES:
        assert (first / filename).read_bytes() != (second / filename).read_bytes()


def test_manifest_hashes_artifacts_not_itself(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    assert ARTIFACT_FILENAMES[-1] == "manifest.json"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {"events.jsonl", "labels.jsonl", "report.json"}
    for filename, metadata in manifest["artifacts"].items():
        assert metadata["sha256"] == sha256_file(output / filename)
        assert metadata["byte_size"] == (output / filename).stat().st_size
    assert validate_dataset_directory(output)["event_count"] == 300


def test_canonical_artifacts_are_lf_terminated_and_compact(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    for filename in ARTIFACT_FILENAMES:
        content = (output / filename).read_bytes()
        assert content.endswith(b"\n")
        assert b"\r" not in content
        assert b": " not in content


def test_event_writer_rejects_arbitrary_mappings(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    with pytest.raises(ArtifactPublishError, match="event_model_invalid") as exc_info:
        write_event_jsonl(
            output,
            [{"raw_payload": "synthetic-sensitive-marker"}],  # type: ignore[list-item]
        )
    assert "synthetic-sensitive-marker" not in str(exc_info.value)
    assert not output.exists()


def test_output_refuses_unsafe_nonempty_and_unknown_overwrite(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    with pytest.raises(ArtifactPublishError, match="unsafe_output_directory"):
        generate_dataset(tiny_config, 1, Path.cwd())

    output = tmp_path / "output"
    generate_dataset(tiny_config, 1, output)
    with pytest.raises(ArtifactPublishError, match="not_empty"):
        generate_dataset(tiny_config, 1, output)
    (output / "unknown.txt").write_text("synthetic marker", encoding="utf-8")
    with pytest.raises(ArtifactPublishError, match="unknown_files"):
        generate_dataset(tiny_config, 1, output, overwrite=True)
    assert (output / "unknown.txt").is_file()


def test_output_refuses_home_root_file_and_symlink_ancestor(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for unsafe in (Path.home(), Path(Path.cwd().anchor)):
        with pytest.raises(ArtifactPublishError, match="unsafe_output_directory"):
            generate_dataset(tiny_config, 1, unsafe)

    file_path = tmp_path / "file-output"
    file_path.write_text("synthetic marker", encoding="utf-8")
    with pytest.raises(ArtifactPublishError, match="not_directory"):
        generate_dataset(tiny_config, 1, file_path)

    symlink_parent = (tmp_path / "linked-parent").absolute()
    output = symlink_parent / "output"
    original_is_symlink = Path.is_symlink

    def is_test_symlink(path: Path) -> bool:
        return path.absolute() == symlink_parent or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_test_symlink)
    with pytest.raises(ArtifactPublishError, match="unsafe_symlinked"):
        generate_dataset(tiny_config, 1, output)


def test_guarded_overwrite_replaces_known_dataset(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    output = tmp_path / "output"
    first = generate_dataset(tiny_config, 1, output)
    second = generate_dataset(tiny_config, 2, output, overwrite=True)
    assert first.dataset_id != second.dataset_id
    assert validate_dataset_directory(output)["seed"] == 2
    assert not any(
        path.name.startswith(".riskloom-simulation-staging-") for path in tmp_path.iterdir()
    )

    empty_output = tmp_path / "empty-output"
    empty_output.mkdir()
    assert generate_dataset(tiny_config, 3, empty_output).event_count == 300


def test_overwrite_requires_a_fully_valid_existing_dataset(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    output = tmp_path / "output"
    generate_dataset(tiny_config, 1, output)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["event_count"] = 1
    write_canonical_json(report_path, report)

    with pytest.raises(ArtifactPublishError, match="existing_dataset_invalid"):
        generate_dataset(tiny_config, 2, output, overwrite=True)
    assert json.loads(report_path.read_text(encoding="utf-8"))["event_count"] == 1


def test_partial_and_noncanonical_artifacts_are_detected(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    partial = tmp_path / "partial"
    generate_dataset(tiny_config, 1, partial)
    (partial / "manifest.json").unlink()
    with pytest.raises(DatasetValidationError, match="artifact_unreadable"):
        validate_dataset_directory(partial)

    noncanonical = tmp_path / "noncanonical"
    generate_dataset(tiny_config, 1, noncanonical)
    report_path = noncanonical / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(DatasetValidationError, match="artifact_not_canonical"):
        validate_dataset_directory(noncanonical)


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("malformed", "existing_manifest_unreadable"),
        ("non_object", "existing_manifest_invalid"),
        ("marker", "existing_manifest_marker_invalid"),
    ],
)
def test_overwrite_rejects_invalid_existing_manifest_safely(
    kind: str,
    error: str,
    tmp_path: Path,
    tiny_config: GeneratorConfig,
) -> None:
    output = tmp_path / "output"
    generate_dataset(tiny_config, 1, output)
    manifest_path = output / "manifest.json"
    if kind == "malformed":
        manifest_path.write_bytes(b"{invalid\n")
    elif kind == "non_object":
        write_canonical_json(manifest_path, [])
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["product"] = "synthetic-other-product"
        write_canonical_json(manifest_path, manifest)
    with pytest.raises(ArtifactPublishError, match=error) as exc_info:
        generate_dataset(tiny_config, 2, output, overwrite=True)
    assert exc_info.value.__cause__ is None


def test_staging_and_publication_failures_are_safe_and_scoped(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def staging_failure(*args, **kwargs):
        del args, kwargs
        raise OSError("synthetic-sensitive-marker")

    monkeypatch.setattr(artifact_module.tempfile, "mkdtemp", staging_failure)
    with pytest.raises(ArtifactPublishError, match="artifact_staging_failed") as exc_info:
        generate_dataset(tiny_config, 1, tmp_path / "staging-failure")
    assert "synthetic-sensitive-marker" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None

    monkeypatch.undo()

    def publication_failure(*args, **kwargs):
        del args, kwargs
        raise OSError("synthetic-sensitive-marker")

    monkeypatch.setattr(artifact_module.os, "replace", publication_failure)
    with pytest.raises(ArtifactPublishError, match="artifact_publication_failed") as exc_info:
        generate_dataset(tiny_config, 1, tmp_path / "publication-failure")
    assert "synthetic-sensitive-marker" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert not any(
        path.name.startswith(".riskloom-simulation-staging-") for path in tmp_path.iterdir()
    )


def test_cleanup_refuses_non_staging_target(
    tmp_path: Path,
    tiny_config: GeneratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected = tmp_path / "must-not-delete"
    unexpected.mkdir()

    def return_unexpected_path(*args, **kwargs) -> str:
        del args, kwargs
        return str(unexpected)

    monkeypatch.setattr(artifact_module.tempfile, "mkdtemp", return_unexpected_path)
    with pytest.raises(ArtifactPublishError, match="unsafe_staging_cleanup_target"):
        generate_dataset(tiny_config, 1, tmp_path / "output")
    assert unexpected.is_dir()
