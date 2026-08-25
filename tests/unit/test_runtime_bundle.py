"""The runtime bundle: what it packages, what it refuses, and what it proves after installing.

These tests never touch the real locked artifacts and never make a network request. Every fixture
is a tiny canonical JSON document written into a temporary tree, with the module's pinned hash table
and serving-binding call patched to match. That keeps the tests about the *mechanism* -- allowlist,
determinism, archive safety, overwrite policy, rollback -- rather than about the contents of files
this gate must not modify.

Two conventions run through the file, both from the security review:

* A rejection test asserts the *specific* identity, not merely that something was refused. An
  archive rejected for the wrong reason is a check that will stop working the moment the allowlist
  changes.
* A rejection test also asserts that nothing was written and nothing was read-modified. "Refused"
  and "refused without side effects" are different guarantees.
"""

import hashlib
import json
import stat
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from riskloom import runtime_bundle
from riskloom.runtime_bundle import (
    ASSET_NAME,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_MEMBERS,
    BUNDLE_SCHEMA_VERSION,
    EVALUATION_MEMBER,
    EXPECTED_SHA256,
    MODEL_DIRECTORY,
    PERMITTED_DOWNLOAD_HOSTS,
    RELEASE_TAG,
    RELEASE_URL,
    STARTUP_MEMBERS,
    SUBMISSION_TAG,
    InstallationResult,
    RuntimeBundleError,
    build_bundle,
    build_manifest,
    download_release_asset,
    install_from_archive,
    install_from_release,
    raw_member_names,
    read_validated_archive,
    verify_installation,
)

CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
EOCD_SIGNATURE = b"PK\x05\x06"
FEATURE_MANIFEST = "artifacts/features/development-v1.1.0-config-bound/manifest.json"

# Captured before the autouse fixture swaps the binding for fixture hashes.
REAL_EXPECTED_SHA256 = runtime_bundle.EXPECTED_SHA256
REAL_EXPECTED_ARCHIVE_SHA256 = runtime_bundle.EXPECTED_ARCHIVE_SHA256

# The archive these fixtures build. Deterministic for the same fixture bytes and module
# constants, exactly as the real bundle is, so it is pinned here the same way. If it goes
# stale the tests fail loudly, which mirrors what production would do.
FIXTURE_ARCHIVE_SHA256 = "6ddaf09cd81f9a17d45098dbdd86c1e890ede26a0c57c73535a82146eeba842d"


def canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# Tiny stand-ins. The real artifacts are never copied into the test suite.
MODEL_ID = "m" * 8
FEATURE_DATASET_ID = "f" * 8

FIXTURES: dict[str, bytes] = {
    f"{MODEL_DIRECTORY}/model.json": canonical({"kind": "model"}),
    f"{MODEL_DIRECTORY}/training_report.json": canonical({"kind": "training_report"}),
    f"{MODEL_DIRECTORY}/manifest.json": canonical({"kind": "model_manifest", "model_id": MODEL_ID}),
    FEATURE_MANIFEST: canonical(
        {"feature_dataset_id": FEATURE_DATASET_ID, "kind": "feature_manifest"}
    ),
    EVALUATION_MEMBER: canonical({"kind": "evaluation"}),
}

FIXTURE_HASHES = {member: digest(payload) for member, payload in FIXTURES.items()}


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module's pinned hash table at the fixtures and stub the binding call."""

    monkeypatch.setattr(runtime_bundle, "EXPECTED_SHA256", dict(FIXTURE_HASHES))
    monkeypatch.setattr(runtime_bundle, "EXPECTED_ARCHIVE_SHA256", FIXTURE_ARCHIVE_SHA256)
    monkeypatch.setattr(runtime_bundle, "load_serving_bundle", lambda **_: object())


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary tree holding the fixture artifacts, treated as the repository root."""

    root = tmp_path / "source"
    for member, payload in FIXTURES.items():
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(runtime_bundle, "REPOSITORY_ROOT", root)
    return root


@pytest.fixture
def archive(source: Path, tmp_path: Path) -> Path:
    return build_bundle(tmp_path / "out" / ASSET_NAME, root=source)


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    root = tmp_path / "clone"
    root.mkdir()
    return root


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under a tree, by relative path, for exact before/after comparison."""

    return {
        str(path.relative_to(root)).replace("\\", "/"): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def rebuild(archive_path: Path, members: dict[str, bytes]) -> Path:
    """Write an archive containing exactly ``members``, bypassing every builder check."""

    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, payload in members.items():
            info = zipfile.ZipInfo(filename=name, date_time=runtime_bundle.ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (0o644 & 0xFFFF) << 16
            handle.writestr(info, payload)
    return archive_path


def members_of(archive_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive_path) as handle:
        return {name: handle.read(name) for name in handle.namelist()}


def refuses(archive_path: Path, identity: str) -> None:
    """Assert the exact rejection identity, so a check cannot pass for the wrong reason.

    The whole-archive pin is neutralised for the duration, by pinning it to the digest of the file
    actually under test. Without that, every mutation below would stop at
    ``archive_hash_mismatch`` and the inner layer -- member names, modes, manifest fields -- would
    never be reached, so a whole class of checks would silently stop being exercised.

    The outer pin is not thereby untested: it has its own tests, which assert that it fires first
    and that nothing downstream of it runs.
    """

    payload = archive_path.read_bytes() if archive_path.is_file() else b""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_bundle, "EXPECTED_ARCHIVE_SHA256", digest(payload))
    try:
        with pytest.raises(RuntimeBundleError) as caught:
            read_validated_archive(archive_path)
    finally:
        monkeypatch.undo()
    assert str(caught.value) == identity, f"expected {identity}, got {caught.value}"


def rebuild_raw(archive_path: Path, payload: bytes) -> Path:
    """Write exact bytes to the archive path, bypassing every writer."""

    archive_path.write_bytes(payload)
    return archive_path


# ------------------------------------------------------------------ the approved member set


def test_the_bundle_carries_exactly_the_five_approved_artifacts() -> None:
    assert len(BUNDLE_MEMBERS) == 5
    assert set(BUNDLE_MEMBERS) == set(EXPECTED_SHA256)
    assert set(BUNDLE_MEMBERS) == set(STARTUP_MEMBERS) | {EVALUATION_MEMBER}


def test_the_allowlist_has_one_source_of_truth_enforced_at_import() -> None:
    """Three collections describe the member set; drift between them must be impossible."""

    assert set(runtime_bundle.EXPECTED_SHA256) == set(BUNDLE_MEMBERS)
    assert len(BUNDLE_MEMBERS) == len(set(BUNDLE_MEMBERS))
    assert BUNDLE_MANIFEST_NAME not in BUNDLE_MEMBERS
    assert {*BUNDLE_MEMBERS, BUNDLE_MANIFEST_NAME} == runtime_bundle.ARCHIVE_NAMES


def test_the_real_pinned_table_is_immutable() -> None:
    """A pinned hash that can be mutated at runtime is not pinned.

    Checked against the value captured at import, because the autouse fixture swaps the binding for
    a plain dict so the rest of the file can pin fixture hashes instead.
    """

    from types import MappingProxyType

    assert isinstance(REAL_EXPECTED_SHA256, MappingProxyType)
    with pytest.raises(TypeError):
        REAL_EXPECTED_SHA256["x"] = "y"  # type: ignore[index]


def test_the_training_report_is_required_to_start() -> None:
    """The defect this gate exists to fix: preflight omitted it, startup requires it."""

    assert f"{MODEL_DIRECTORY}/training_report.json" in STARTUP_MEMBERS


def test_the_evaluation_is_bundled_but_is_not_a_startup_requirement() -> None:
    assert EVALUATION_MEMBER in BUNDLE_MEMBERS
    assert EVALUATION_MEMBER not in STARTUP_MEMBERS


def test_no_dataset_or_event_level_artifact_is_ever_a_member() -> None:
    """Events, labels, feature rows and reports must never be distributed."""

    for member in BUNDLE_MEMBERS:
        assert member.endswith(".json")
        assert "features.jsonl" not in member
        assert "labels" not in member
        assert "events" not in member
        assert not member.endswith("report.json") or member.endswith("training_report.json")


# ------------------------------------------------------------------ runtime vs submission tag


def test_the_runtime_release_tag_is_separate_from_the_submission_tag() -> None:
    """The source snapshot may be re-tagged without invalidating a published runtime asset."""

    assert RELEASE_TAG == "v1.0.3-runtime"
    assert SUBMISSION_TAG == "v1.0.3-submission"
    assert RELEASE_TAG != SUBMISSION_TAG


def test_the_download_url_is_built_from_the_runtime_tag_only() -> None:
    expected = (
        f"https://github.com/XxSURYANSHxX/riskloom/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
    )
    assert expected == RELEASE_URL
    assert "v1.0.3-runtime" in RELEASE_URL
    assert SUBMISSION_TAG not in RELEASE_URL
    assert ASSET_NAME == "riskloom-runtime-artifacts.zip"


def test_the_bundle_manifest_records_the_runtime_tag(archive: Path) -> None:
    manifest = json.loads(members_of(archive)[BUNDLE_MANIFEST_NAME])
    assert manifest["release_tag"] == "v1.0.3-runtime"


def test_a_bundle_declaring_the_submission_tag_is_refused(archive: Path) -> None:
    """Nothing may assume the two tags are interchangeable."""

    payloads = members_of(archive)
    manifest = json.loads(payloads[BUNDLE_MANIFEST_NAME])
    manifest["release_tag"] = SUBMISSION_TAG
    payloads[BUNDLE_MANIFEST_NAME] = canonical(manifest)
    refuses(rebuild(archive, payloads), "runtime_bundle_manifest_release_invalid")


def test_no_environment_variable_or_configuration_can_redirect_the_download() -> None:
    """The URL is a source constant, and nothing in the module can read an override.

    Asserted structurally rather than by reloading the module under a poisoned environment:
    reloading would reset the patched state for every test that follows it. If the module cannot
    read the environment or any settings object at all, no override can exist.
    """

    source = Path(runtime_bundle.__file__).read_text(encoding="utf-8")
    for forbidden in ("os.environ", "getenv", "Settings", "load_dotenv", "config."):
        assert forbidden not in source, f"{forbidden} would allow an override"
    assert source.count("RELEASE_URL = ") == 1
    assert "https://github.com/XxSURYANSHxX/riskloom/releases/download/" in source


# ------------------------------------------------------------------ deterministic archive


def test_two_builds_produce_byte_identical_archives(source: Path, tmp_path: Path) -> None:
    first = build_bundle(tmp_path / "a" / ASSET_NAME, root=source)
    second = build_bundle(tmp_path / "b" / ASSET_NAME, root=source)
    assert first.read_bytes() == second.read_bytes()


def test_archive_metadata_is_fixed_not_inherited_from_the_build_machine(archive: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        assert handle.comment == b""
    assert [info.filename for info in infos] == [*sorted(BUNDLE_MEMBERS), BUNDLE_MANIFEST_NAME]
    for info in infos:
        assert info.date_time == runtime_bundle.ZIP_TIMESTAMP
        assert info.compress_type == zipfile.ZIP_STORED
        assert (info.external_attr >> 16) & 0o7777 == 0o644
        assert info.comment == b""
        assert info.extra == b""
        assert info.flag_bits & 0x8 == 0, "no data descriptor"


def test_the_internal_manifest_is_canonical_and_omits_its_own_hash(archive: Path) -> None:
    payload = members_of(archive)[BUNDLE_MANIFEST_NAME]
    manifest = json.loads(payload)
    assert payload == canonical(manifest)
    assert manifest["product"] == "RiskLoom"
    assert manifest["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["members"] == sorted(BUNDLE_MEMBERS)
    assert BUNDLE_MANIFEST_NAME not in manifest["artifacts"]


def test_the_manifest_carries_no_machine_or_time_dependent_value(archive: Path) -> None:
    """Determinism is only credible if nothing environmental leaks into the manifest."""

    import getpass
    import platform
    import socket

    rendered = members_of(archive)[BUNDLE_MANIFEST_NAME].decode()
    for leak in (socket.gethostname(), platform.node(), getpass.getuser(), str(Path.home())):
        if leak:
            assert leak not in rendered
    for key in ("timestamp", "created_at", "hostname", "path", "user", "locale"):
        assert key not in rendered


def test_the_manifest_records_exact_size_and_hash_for_every_artifact() -> None:
    manifest = build_manifest(dict(FIXTURES))
    for member, payload in FIXTURES.items():
        entry = manifest["artifacts"][member]
        assert entry == {"byte_size": len(payload), "sha256": digest(payload)}
    assert manifest["source"] == {
        "feature_dataset_id": FEATURE_DATASET_ID,
        "model_id": MODEL_ID,
    }


# ------------------------------------------------------------------ build refusals


def test_a_missing_source_artifact_is_refused(source: Path, tmp_path: Path) -> None:
    (source / f"{MODEL_DIRECTORY}/training_report.json").unlink()
    with pytest.raises(RuntimeBundleError, match="source_missing"):
        build_bundle(tmp_path / "out" / ASSET_NAME, root=source)
    assert not (tmp_path / "out" / ASSET_NAME).exists()


def test_a_source_hash_mismatch_is_refused(source: Path, tmp_path: Path) -> None:
    (source / EVALUATION_MEMBER).write_bytes(canonical({"kind": "tampered"}))
    with pytest.raises(RuntimeBundleError, match="source_hash_mismatch"):
        build_bundle(tmp_path / "out" / ASSET_NAME, root=source)


def test_noncanonical_source_json_is_refused(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-canonical JSON is refused as such, before the hash check can mask it."""

    pretty = json.dumps({"kind": "model"}, indent=2).encode()
    (source / f"{MODEL_DIRECTORY}/model.json").write_bytes(pretty)
    adjusted = dict(FIXTURE_HASHES)
    adjusted[f"{MODEL_DIRECTORY}/model.json"] = digest(pretty)
    monkeypatch.setattr(runtime_bundle, "EXPECTED_SHA256", adjusted)
    with pytest.raises(RuntimeBundleError) as caught:
        build_bundle(tmp_path / "out" / ASSET_NAME, root=source)
    assert str(caught.value) == "runtime_bundle_json_not_canonical"


def test_an_empty_source_artifact_is_refused(source: Path, tmp_path: Path) -> None:
    (source / EVALUATION_MEMBER).write_bytes(b"")
    with pytest.raises(RuntimeBundleError, match="source_empty"):
        build_bundle(tmp_path / "out" / ASSET_NAME, root=source)


def test_a_directory_in_place_of_an_artifact_is_refused(source: Path, tmp_path: Path) -> None:
    path = source / EVALUATION_MEMBER
    path.unlink()
    path.mkdir()
    with pytest.raises(RuntimeBundleError, match="source_not_a_file"):
        build_bundle(tmp_path / "out" / ASSET_NAME, root=source)


def test_an_existing_output_archive_is_never_overwritten(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / ASSET_NAME
    build_bundle(output, root=source)
    before = output.read_bytes()
    with pytest.raises(RuntimeBundleError, match="output_exists"):
        build_bundle(output, root=source)
    assert output.read_bytes() == before


def test_an_unsafe_output_path_is_refused(source: Path) -> None:
    with pytest.raises(RuntimeBundleError, match="output_unsafe"):
        build_bundle(Path.home(), root=source)
    with pytest.raises(RuntimeBundleError, match="output_unsafe"):
        build_bundle(source, root=source)


@pytest.mark.parametrize(
    "relative",
    [
        f"{MODEL_DIRECTORY}/bundle.zip",
        "artifacts/features/development-v1.1.0-config-bound/bundle.zip",
        "artifacts/evaluations/development/nested/bundle.zip",
    ],
)
def test_output_inside_a_protected_artifact_directory_is_refused(
    source: Path, relative: str
) -> None:
    """A fourth file in the model directory makes the strict loader reject the whole directory."""

    before = snapshot(source)
    with pytest.raises(RuntimeBundleError, match="output_inside_artifact_tree"):
        build_bundle(source / relative, root=source)
    assert snapshot(source) == before


def test_output_onto_a_source_artifact_is_refused(source: Path) -> None:
    """Asserts the identity that actually fires, not merely that something was refused.

    Every approved member lives inside a protected directory, so the directory rule is reached
    first and ``output_is_source_artifact`` cannot currently fire. That second guard is retained as
    defence in depth for a future member outside those directories; pinning the real identity here
    means the day it becomes reachable, this test says so instead of quietly continuing to pass.
    """

    before = snapshot(source)
    with pytest.raises(RuntimeBundleError) as caught:
        build_bundle(source / EVALUATION_MEMBER, root=source)
    assert str(caught.value) == "runtime_bundle_output_inside_artifact_tree"
    assert snapshot(source) == before


def test_the_source_artifact_guard_is_currently_unreachable_and_that_is_recorded() -> None:
    """The condition under which the second guard would start firing.

    Not a defect, but not something to leave implicit either: an unreachable check looks like
    protection it is not currently providing.
    """

    outside = [
        member
        for member in BUNDLE_MEMBERS
        if not any(member.startswith(f"{d}/") for d in runtime_bundle.PROTECTED_DIRECTORIES)
    ]
    assert outside == [], (
        "a member now sits outside every protected directory, so output_is_source_artifact "
        "has become reachable and its own test should assert that identity directly"
    )


def test_building_does_not_modify_the_source_artifacts(source: Path, tmp_path: Path) -> None:
    before = snapshot(source)
    build_bundle(tmp_path / "out" / ASSET_NAME, root=source)
    assert snapshot(source) == before


def test_a_failed_build_leaves_no_partial_archive(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure forced *after* the output directory and the temporary archive exist.

    The previous version tampered a source artifact, which fails in ``_read_approved_artifact``
    before ``mkdir`` ever runs -- so the output directory did not exist and the ``.partial``
    assertion was guarded by an ``if`` that made it vacuously true. It asserted nothing.
    """

    output = tmp_path / "out" / ASSET_NAME
    unrelated = tmp_path / "out" / "keep.txt"
    output.parent.mkdir(parents=True)
    unrelated.write_text("untouched")

    # Fails after the ZIP has been written to its staging name, which is the state the cleanup
    # exists for.
    monkeypatch.setattr(runtime_bundle, "EXPECTED_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(RuntimeBundleError) as caught:
        build_bundle(output, root=source)
    assert str(caught.value) == "runtime_bundle_build_hash_mismatch"

    assert not output.exists()
    assert list((tmp_path / "out").glob("*.partial")) == []
    assert list((tmp_path / "out").glob(".*")) == []
    assert unrelated.read_text() == "untouched"


def test_a_build_hash_mismatch_never_replaces_an_existing_output(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing destination is refused before the build, and survives untouched."""

    output = tmp_path / "out" / ASSET_NAME
    output.parent.mkdir(parents=True)
    output.write_bytes(b"an existing file")

    monkeypatch.setattr(runtime_bundle, "EXPECTED_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(RuntimeBundleError) as caught:
        build_bundle(output, root=source)
    assert str(caught.value) == "runtime_bundle_output_exists"
    assert output.read_bytes() == b"an existing file"


def test_the_build_verifies_the_completed_archive_against_the_pin(
    source: Path, tmp_path: Path
) -> None:
    """The happy path: a build that reproduces the pinned container publishes normally."""

    built = build_bundle(tmp_path / "ok" / ASSET_NAME, root=source)
    assert digest(built.read_bytes()) == FIXTURE_ARCHIVE_SHA256


# ------------------------------------------------------------------ raw central-directory parser


def test_the_raw_parser_returns_the_stored_names(archive: Path) -> None:
    names = raw_member_names(archive.read_bytes())
    assert sorted(names) == sorted({*BUNDLE_MEMBERS, BUNDLE_MANIFEST_NAME})


def test_the_raw_parser_is_not_fooled_by_a_signature_inside_member_data(tmp_path: Path) -> None:
    """The defect that motivated replacing signature scanning with an EOCD-anchored parse.

    A hostile archive is under no obligation to hold canonical JSON -- payload validation happens
    much later -- so a member's stored data can contain a forged central-directory record naming a
    traversal path. A parser that scans for the signature reads those attacker-chosen bytes as a
    real record. The EOCD-anchored parse must see exactly the two genuine members.
    """

    decoy = bytearray(CENTRAL_DIRECTORY_SIGNATURE)
    decoy += b"\x00" * 24
    decoy += len(b"../../evil.json").to_bytes(2, "little")  # filename length sits at offset 28
    decoy += b"\x00" * 16
    decoy += b"../../evil.json"

    hostile = tmp_path / "hostile.zip"
    with zipfile.ZipFile(hostile, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("a.json", bytes(decoy))
        handle.writestr("b.json", b"plain")

    payload = hostile.read_bytes()
    assert payload.count(CENTRAL_DIRECTORY_SIGNATURE) == 3, "one decoy plus two real records"

    names = raw_member_names(payload)
    assert names == ["a.json", "b.json"], names
    assert "../../evil.json" not in names


def test_trailing_garbage_after_the_eocd_is_refused(archive: Path) -> None:
    archive.write_bytes(archive.read_bytes() + b"appended")
    refuses(archive, "runtime_bundle_archive_eocd_invalid")


def test_an_archive_comment_is_refused(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "commented.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, payload in FIXTURES.items():
            handle.writestr(name, payload)
        handle.writestr(BUNDLE_MANIFEST_NAME, canonical(build_manifest(dict(FIXTURES))))
        handle.comment = b"unexpected"
    refuses(output, "runtime_bundle_archive_comment_unsupported")


def test_a_zip64_archive_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    raw[eocd:eocd] = b"PK\x06\x07" + b"\x00" * 16
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_zip64_unsupported")


def test_a_multi_disk_archive_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    raw[eocd + 4 : eocd + 6] = (1).to_bytes(2, "little")
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_multi_disk_unsupported")


def test_a_record_count_disagreement_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    raw[eocd + 8 : eocd + 10] = (99).to_bytes(2, "little")
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_record_count_mismatch")


def test_a_directory_offset_outside_the_archive_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    raw[eocd + 16 : eocd + 20] = (0xFFFFFF).to_bytes(4, "little")
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_directory_out_of_bounds")


def test_a_directory_size_that_does_not_reach_the_eocd_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    size = int.from_bytes(raw[eocd + 12 : eocd + 16], "little")
    raw[eocd + 12 : eocd + 16] = (size - 1).to_bytes(4, "little")
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_directory_extent_invalid")


def test_a_corrupt_directory_record_signature_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    offset = int.from_bytes(raw[eocd + 16 : eocd + 20], "little")
    raw[offset : offset + 4] = b"XXXX"
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_directory_signature_invalid")


def test_an_archive_shorter_than_an_eocd_is_refused(tmp_path: Path) -> None:
    stub = tmp_path / "stub.zip"
    stub.write_bytes(b"not a zip at all")
    refuses(stub, "runtime_bundle_archive_truncated")


def test_a_file_without_an_eocd_is_refused(tmp_path: Path) -> None:
    stub = tmp_path / "long.zip"
    stub.write_bytes(b"x" * 512)
    refuses(stub, "runtime_bundle_archive_eocd_invalid")


def test_an_empty_archive_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.zip"
    empty.write_bytes(b"")
    refuses(empty, "runtime_bundle_archive_empty")


def test_a_missing_archive_is_refused(tmp_path: Path) -> None:
    refuses(tmp_path / "absent.zip", "runtime_bundle_archive_missing")


# ------------------------------------------------------------------ member-level rejection


def test_an_unknown_member_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    payloads[f"{MODEL_DIRECTORY}/extra.json"] = canonical({"kind": "extra"})
    refuses(rebuild(archive, payloads), "runtime_bundle_member_unknown")


def test_a_missing_member_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    del payloads[EVALUATION_MEMBER]
    refuses(rebuild(archive, payloads), "runtime_bundle_member_missing")


def test_a_duplicate_member_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, payload in payloads.items():
            handle.writestr(name, payload)
        handle.writestr(EVALUATION_MEMBER, payloads[EVALUATION_MEMBER])
    refuses(archive, "runtime_bundle_member_duplicate")


@pytest.mark.parametrize(
    ("name", "identity"),
    [
        ("../escape.json", "runtime_bundle_member_traversal"),
        ("artifacts/../../escape.json", "runtime_bundle_member_traversal"),
        ("/etc/passwd", "runtime_bundle_member_absolute"),
        ("./artifacts/models/development/model.json", "runtime_bundle_member_absolute"),
        ("C:/Windows/system32/x.json", "runtime_bundle_member_absolute"),
        ("artifacts//models/x.json", "runtime_bundle_member_name_invalid"),
        ("artifacts/./models/x.json", "runtime_bundle_member_name_invalid"),
        ("artifacts/models/x.json ", "runtime_bundle_member_name_invalid"),
        ("artifacts/models/\tx.json", "runtime_bundle_member_name_invalid"),
    ],
)
def test_unsafe_member_names_are_refused_for_the_right_reason(
    archive: Path, name: str, identity: str
) -> None:
    payloads = members_of(archive)
    payloads[name] = canonical({"kind": "hostile"})
    refuses(rebuild(archive, payloads), identity)


def test_a_backslash_member_name_is_refused(archive: Path) -> None:
    """``ZipInfo.__init__`` rewrites ``os.sep`` to ``/`` on Windows.

    So the hostile name has to be assigned after construction -- which is exactly how a
    hand-crafted archive built on another platform would carry it. Without the raw parser this
    archive is refused as an *unknown member*, which is the wrong reason.
    """

    payloads = members_of(archive)
    archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, payload in payloads.items():
            handle.writestr(name, payload)
        info = zipfile.ZipInfo("placeholder", date_time=runtime_bundle.ZIP_TIMESTAMP)
        info.filename = "artifacts\\models\\development\\evil.json"
        info.compress_type = zipfile.ZIP_STORED
        handle.writestr(info, canonical({"kind": "hostile"}))
    refuses(archive, "runtime_bundle_member_backslash")


def test_a_symlink_member_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, payload in payloads.items():
            if name == EVALUATION_MEMBER:
                info = zipfile.ZipInfo(filename=name, date_time=runtime_bundle.ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                handle.writestr(info, "../../../etc/passwd")
                continue
            handle.writestr(name, payload)
    refuses(archive, "runtime_bundle_member_symlink")


def test_a_device_member_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for name, payload in payloads.items():
            info = zipfile.ZipInfo(filename=name, date_time=runtime_bundle.ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            mode = stat.S_IFCHR if name == EVALUATION_MEMBER else stat.S_IFREG
            info.external_attr = (mode | 0o644) << 16
            handle.writestr(info, payload)
    refuses(archive, "runtime_bundle_member_not_regular")


def test_a_directory_entry_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr(zipfile.ZipInfo(f"{MODEL_DIRECTORY}/"), b"")
        for name, payload in payloads.items():
            handle.writestr(name, payload)
    refuses(archive, "runtime_bundle_member_directory")


def test_an_encrypted_member_is_refused(archive: Path) -> None:
    """``writestr`` clears ``flag_bits``, so the bit is set where ``infolist`` reads it from."""

    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    offset = int.from_bytes(raw[eocd + 16 : eocd + 20], "little")
    patched = 0
    cursor = offset
    while cursor < eocd and raw[cursor : cursor + 4] == CENTRAL_DIRECTORY_SIGNATURE:
        raw[cursor + 8] |= 0x01  # general-purpose bit flag, bit 0 = encrypted
        name_length = int.from_bytes(raw[cursor + 28 : cursor + 30], "little")
        extra = int.from_bytes(raw[cursor + 30 : cursor + 32], "little")
        comment = int.from_bytes(raw[cursor + 32 : cursor + 34], "little")
        cursor += 46 + name_length + extra + comment
        patched += 1
    assert patched == len(BUNDLE_MEMBERS) + 1
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_member_encrypted")


def test_an_unsupported_compression_method_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    archive.unlink()
    with zipfile.ZipFile(archive, "w") as handle:
        for name, payload in payloads.items():
            method = zipfile.ZIP_DEFLATED if name == EVALUATION_MEMBER else zipfile.ZIP_STORED
            info = zipfile.ZipInfo(filename=name, date_time=runtime_bundle.ZIP_TIMESTAMP)
            info.compress_type = method
            handle.writestr(info, payload)
    refuses(archive, "runtime_bundle_member_compression_unsupported")


def test_an_oversized_member_is_refused(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_bundle, "MAXIMUM_MEMBER_BYTES", 16)
    refuses(archive, "runtime_bundle_member_too_large")


def test_an_oversized_total_is_refused(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_bundle, "MAXIMUM_TOTAL_BYTES", 8)
    refuses(archive, "runtime_bundle_total_too_large")


def test_an_oversized_archive_is_refused(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_bundle, "MAXIMUM_ARCHIVE_BYTES", 8)
    refuses(archive, "runtime_bundle_archive_too_large")


def test_a_corrupt_member_crc_is_refused(archive: Path) -> None:
    """zipfile checks the CRC at EOF; a flipped data byte must not reach the destination."""

    raw = bytearray(archive.read_bytes())
    # First local header: 30 fixed bytes + filename, then the stored data.
    name_length = int.from_bytes(raw[26:28], "little")
    extra_length = int.from_bytes(raw[28:30], "little")
    data = 30 + name_length + extra_length
    raw[data] ^= 0xFF
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_corrupt")


def test_a_local_header_name_disagreement_is_refused(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    name_length = int.from_bytes(raw[26:28], "little")
    raw[30 : 30 + 4] = b"zzzz"
    assert name_length > 4
    archive.write_bytes(bytes(raw))
    refuses(archive, "runtime_bundle_archive_corrupt")


# ------------------------------------------------------------------ manifest tampering


def _with_manifest(archive: Path, mutate: Any) -> Path:
    payloads = members_of(archive)
    manifest = json.loads(payloads[BUNDLE_MANIFEST_NAME])
    mutate(manifest)
    payloads[BUNDLE_MANIFEST_NAME] = canonical(manifest)
    return rebuild(archive, payloads)


@pytest.mark.parametrize(
    ("key", "value", "identity"),
    [
        ("product", "SomethingElse", "runtime_bundle_manifest_product_invalid"),
        ("artifact_type", "other", "runtime_bundle_manifest_artifact_type_invalid"),
        ("bundle_schema_version", "2.0.0", "runtime_bundle_manifest_schema_invalid"),
        ("release_tag", "v9.9.9-other", "runtime_bundle_manifest_release_invalid"),
    ],
)
def test_a_wrong_manifest_identity_is_refused(
    archive: Path, key: str, value: str, identity: str
) -> None:
    refuses(_with_manifest(archive, lambda m: m.__setitem__(key, value)), identity)


def test_an_unknown_manifest_key_is_refused(archive: Path) -> None:
    """The manifest schema is closed; a bundle may not smuggle extra fields past validation."""

    refuses(
        _with_manifest(archive, lambda m: m.__setitem__("extra", "x")),
        "runtime_bundle_manifest_schema_invalid",
    )


def test_a_wrong_allowlist_is_refused(archive: Path) -> None:
    refuses(
        _with_manifest(archive, lambda m: m.__setitem__("members", sorted(STARTUP_MEMBERS))),
        "runtime_bundle_manifest_members_invalid",
    )


def test_a_manifest_source_identity_mismatch_is_refused(archive: Path) -> None:
    """The declared source must match the manifests actually shipped."""

    refuses(
        _with_manifest(archive, lambda m: m["source"].__setitem__("model_id", "z" * 8)),
        "runtime_bundle_manifest_source_invalid",
    )


def test_a_wrong_declared_size_is_refused(archive: Path) -> None:
    refuses(
        _with_manifest(
            archive, lambda m: m["artifacts"][EVALUATION_MEMBER].__setitem__("byte_size", 1)
        ),
        "runtime_bundle_declared_size_mismatch",
    )


def test_a_wrong_declared_hash_is_refused(archive: Path) -> None:
    refuses(
        _with_manifest(
            archive, lambda m: m["artifacts"][EVALUATION_MEMBER].__setitem__("sha256", "0" * 64)
        ),
        "runtime_bundle_declared_hash_mismatch",
    )


def test_a_noncanonical_manifest_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    payloads[BUNDLE_MANIFEST_NAME] = json.dumps(
        json.loads(payloads[BUNDLE_MANIFEST_NAME]), indent=2
    ).encode()
    refuses(rebuild(archive, payloads), "runtime_bundle_json_not_canonical")


def test_a_consistently_re_signed_tampered_artifact_is_refused(archive: Path) -> None:
    """The strongest case: payload and manifest agree, but neither matches the pinned value."""

    payloads = members_of(archive)
    forged = canonical({"kind": "forged"})
    payloads[EVALUATION_MEMBER] = forged
    manifest = json.loads(payloads[BUNDLE_MANIFEST_NAME])
    manifest["artifacts"][EVALUATION_MEMBER] = {
        "byte_size": len(forged),
        "sha256": digest(forged),
    }
    payloads[BUNDLE_MANIFEST_NAME] = canonical(manifest)
    refuses(rebuild(archive, payloads), "runtime_bundle_artifact_hash_mismatch")


def test_a_tampered_artifact_of_identical_length_is_refused(archive: Path) -> None:
    """Same byte count, different bytes: the size check cannot catch this one, so the hash must."""

    forged = canonical({"kind": "evaluatiox"})
    assert len(forged) == len(FIXTURES[EVALUATION_MEMBER])
    assert forged != FIXTURES[EVALUATION_MEMBER]
    payloads = members_of(archive)
    payloads[EVALUATION_MEMBER] = forged
    refuses(rebuild(archive, payloads), "runtime_bundle_declared_hash_mismatch")


def test_noncanonical_artifact_json_inside_the_archive_is_refused(archive: Path) -> None:
    payloads = members_of(archive)
    payloads[EVALUATION_MEMBER] = json.dumps({"kind": "evaluation"}, indent=2).encode()
    refuses(rebuild(archive, payloads), "runtime_bundle_json_not_canonical")


def test_validation_writes_nothing_to_disk(archive: Path, destination: Path) -> None:
    payloads = members_of(archive)
    payloads[EVALUATION_MEMBER] = canonical({"kind": "tampered"})
    rebuild(archive, payloads)
    before = snapshot(destination)
    with pytest.raises(RuntimeBundleError):
        read_validated_archive(archive)
    assert snapshot(destination) == before


# ------------------------------------------------------------------ installation


def test_a_clean_installation_writes_every_approved_file(archive: Path, destination: Path) -> None:
    result = install_from_archive(archive, root=destination)
    assert set(result.installed) == set(BUNDLE_MEMBERS)
    assert result.already_present == ()
    assert result.evaluation_installed is True
    for member, payload in FIXTURES.items():
        assert (destination / member).read_bytes() == payload


def test_installation_is_idempotent(archive: Path, destination: Path) -> None:
    install_from_archive(archive, root=destination)
    after_first = snapshot(destination)
    second = install_from_archive(archive, root=destination)
    assert second.installed == ()
    assert set(second.already_present) == set(BUNDLE_MEMBERS)
    assert snapshot(destination) == after_first


def test_different_existing_content_is_never_overwritten(archive: Path, destination: Path) -> None:
    path = destination / EVALUATION_MEMBER
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical({"kind": "someone_elses"}))
    before = snapshot(destination)
    with pytest.raises(RuntimeBundleError, match="destination_conflict"):
        install_from_archive(archive, root=destination)
    assert snapshot(destination) == before, "a conflict must abort before any file is written"


def test_a_conflict_aborts_before_any_member_is_published(archive: Path, destination: Path) -> None:
    """Every destination is classified first, so a late conflict cannot leave earlier files."""

    path = destination / f"{MODEL_DIRECTORY}/model.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical({"kind": "different"}))
    before = snapshot(destination)
    with pytest.raises(RuntimeBundleError, match="destination_conflict"):
        install_from_archive(archive, root=destination)
    assert snapshot(destination) == before


def test_a_mix_of_identical_and_missing_destinations_installs_only_the_missing(
    archive: Path, destination: Path
) -> None:
    path = destination / EVALUATION_MEMBER
    path.parent.mkdir(parents=True)
    path.write_bytes(FIXTURES[EVALUATION_MEMBER])
    result = install_from_archive(archive, root=destination)
    assert result.already_present == (EVALUATION_MEMBER,)
    assert set(result.installed) == set(BUNDLE_MEMBERS) - {EVALUATION_MEMBER}


def test_unrelated_files_outside_the_strict_directories_are_preserved(
    archive: Path, destination: Path
) -> None:
    keep = destination / "artifacts/evaluations/development/manifest.json"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(canonical({"kind": "evaluation_manifest"}))
    other = destination / "notes.txt"
    other.write_text("unrelated")

    install_from_archive(archive, root=destination)
    assert keep.read_bytes() == canonical({"kind": "evaluation_manifest"})
    assert other.read_text() == "unrelated"


def test_an_unknown_file_in_the_strict_model_directory_is_refused(
    archive: Path, destination: Path
) -> None:
    """``load_locked_model`` requires that directory to hold exactly its three artifacts."""

    stray = destination / MODEL_DIRECTORY / "notes.json"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(canonical({"kind": "stray"}))
    before = snapshot(destination)
    with pytest.raises(RuntimeBundleError, match="model_directory_has_unknown_files"):
        install_from_archive(archive, root=destination)
    assert snapshot(destination) == before


def test_a_startup_only_installation_omits_the_evaluation(archive: Path, destination: Path) -> None:
    result = install_from_archive(archive, root=destination, require_evaluation=False)
    assert set(result.installed) == set(STARTUP_MEMBERS)
    assert result.evaluation_installed is False
    assert not (destination / EVALUATION_MEMBER).exists()


def test_installation_leaves_no_staging_directory_behind(archive: Path, destination: Path) -> None:
    install_from_archive(archive, root=destination)
    assert not [p for p in destination.iterdir() if p.name.startswith(".riskloom-bundle-staging-")]


# ------------------------------------------------------------------ rollback


@pytest.mark.parametrize("failing_step", [1, 2, 3, 4, 5])
def test_a_publication_failure_rolls_the_destination_back_exactly(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch, failing_step: int
) -> None:
    """Failure injection at every publication step, proving an exact return to pre-install state.

    This is the gap the review found: without rollback, a failure on the third file left the first
    two installed, so a retry met a half-populated tree.
    """

    before = snapshot(destination)
    real_replace = runtime_bundle.os.replace
    calls = {"n": 0}

    def failing_replace(a: Any, b: Any) -> None:
        calls["n"] += 1
        if calls["n"] == failing_step:
            raise OSError("injected publication failure")
        real_replace(a, b)

    monkeypatch.setattr(runtime_bundle.os, "replace", failing_replace)
    with pytest.raises(RuntimeBundleError, match="publication_failed"):
        install_from_archive(archive, root=destination)

    assert snapshot(destination) == before, f"step {failing_step} left residue"
    assert not [p for p in destination.iterdir() if p.name.startswith(".riskloom-bundle-staging-")]


def test_rollback_never_removes_a_pre_existing_identical_file(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that was already correct was not created by this attempt and must survive."""

    kept = destination / EVALUATION_MEMBER
    kept.parent.mkdir(parents=True)
    kept.write_bytes(FIXTURES[EVALUATION_MEMBER])
    before = snapshot(destination)

    real_replace = runtime_bundle.os.replace
    calls = {"n": 0}

    def failing_replace(a: Any, b: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected")
        real_replace(a, b)

    monkeypatch.setattr(runtime_bundle.os, "replace", failing_replace)
    with pytest.raises(RuntimeBundleError, match="publication_failed"):
        install_from_archive(archive, root=destination)

    assert kept.read_bytes() == FIXTURES[EVALUATION_MEMBER]
    assert snapshot(destination) == before


def test_installation_after_a_rolled_back_failure_succeeds(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback is only useful if the retry is clean."""

    real_replace = runtime_bundle.os.replace
    calls = {"n": 0}

    def failing_replace(a: Any, b: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("injected")
        real_replace(a, b)

    monkeypatch.setattr(runtime_bundle.os, "replace", failing_replace)
    with pytest.raises(RuntimeBundleError):
        install_from_archive(archive, root=destination)
    monkeypatch.setattr(runtime_bundle.os, "replace", real_replace)

    result = install_from_archive(archive, root=destination)
    assert set(result.installed) == set(BUNDLE_MEMBERS)


# ------------------------------------------------------------------ serving binding


def test_installation_runs_the_ordinary_serving_validation(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(runtime_bundle, "load_serving_bundle", record)
    install_from_archive(archive, root=destination)

    assert len(calls) == 1
    assert calls[0]["model_directory"] == destination / MODEL_DIRECTORY
    assert calls[0]["feature_manifest_path"] == destination / FEATURE_MANIFEST
    assert calls[0]["feature_config_path"] == destination / "configs/features/default.json"
    assert calls[0]["modeling_config_path"] == destination / "configs/modeling/default.json"


def test_a_failing_serving_binding_is_reported(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from riskloom.serving.model_host import ServingBundleError

    def refuse(**_: Any) -> object:
        raise ServingBundleError("serving_locked_model_invalid")

    monkeypatch.setattr(runtime_bundle, "load_serving_bundle", refuse)
    with pytest.raises(RuntimeBundleError, match="binding_failed:serving_locked_model_invalid"):
        install_from_archive(archive, root=destination)


def test_the_binding_check_does_not_require_the_evaluation(
    archive: Path, destination: Path
) -> None:
    """The service starts without it; only the complete install verifies its hash."""

    install_from_archive(archive, root=destination, require_evaluation=False)
    verify_installation(root=destination, require_evaluation=False)
    with pytest.raises(RuntimeBundleError, match="installed_missing"):
        verify_installation(root=destination, require_evaluation=True)


def test_verification_detects_a_corrupted_installed_artifact(
    archive: Path, destination: Path
) -> None:
    install_from_archive(archive, root=destination)
    (destination / EVALUATION_MEMBER).write_bytes(canonical({"kind": "corrupted"}))
    with pytest.raises(RuntimeBundleError, match="installed_hash_mismatch"):
        verify_installation(root=destination, require_evaluation=True)


def test_verification_detects_a_missing_installed_artifact(
    archive: Path, destination: Path
) -> None:
    install_from_archive(archive, root=destination)
    (destination / f"{MODEL_DIRECTORY}/training_report.json").unlink()
    with pytest.raises(RuntimeBundleError, match="installed_missing"):
        verify_installation(root=destination)


# ------------------------------------------------------------------ download safety


def test_the_downloader_accepts_no_caller_supplied_url() -> None:
    """The only way to widen the target would be a URL parameter. There is none."""

    import inspect

    assert set(inspect.signature(download_release_asset).parameters) == {"destination", "client"}


def test_the_download_requests_only_the_fixed_url(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"payload")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    destination = tmp_path / ASSET_NAME
    download_release_asset(destination, client=client)
    assert seen == [RELEASE_URL]
    assert destination.read_bytes() == b"payload"


def test_a_redirect_to_a_permitted_github_host_is_followed(tmp_path: Path) -> None:
    seen: list[str] = []
    target = "https://objects.githubusercontent.com/some/signed/path?token=abc"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == RELEASE_URL:
            return httpx.Response(302, headers={"location": target})
        return httpx.Response(200, content=b"payload")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    download_release_asset(tmp_path / ASSET_NAME, client=client)
    assert seen == [RELEASE_URL, target], "a signed query string must be permitted"


@pytest.mark.parametrize(
    ("location", "identity"),
    [
        ("https://evil.example/payload.zip", "runtime_bundle_download_host_invalid"),
        ("http://github.com/payload.zip", "runtime_bundle_download_scheme_invalid"),
        (
            "https://github.com.evil.example/payload.zip",
            "runtime_bundle_download_host_invalid",
        ),
        (
            "https://evil.githubusercontent.com/payload.zip",
            "runtime_bundle_download_host_invalid",
        ),
        ("https://user:pw@github.com/payload.zip", "runtime_bundle_download_userinfo_rejected"),
        ("https://github.com:8443/payload.zip", "runtime_bundle_download_port_invalid"),
        ("https://github.com/payload.zip#frag", "runtime_bundle_download_fragment_rejected"),
    ],
)
def test_an_unsafe_redirect_target_is_refused(tmp_path: Path, location: str, identity: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": location})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    destination = tmp_path / ASSET_NAME
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(destination, client=client)
    assert str(caught.value) == identity
    assert not destination.exists()


def test_the_permitted_host_set_is_an_exact_list_not_a_suffix_match() -> None:
    expected = {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
    assert expected == PERMITTED_DOWNLOAD_HOSTS


def test_a_redirect_loop_is_bounded(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(302, headers={"location": RELEASE_URL})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError, match="too_many_redirects"):
        download_release_asset(tmp_path / ASSET_NAME, client=client)
    assert attempts["n"] <= runtime_bundle.MAXIMUM_REDIRECTS + 1


def test_a_declared_content_length_over_the_cap_is_refused_before_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_bundle, "MAXIMUM_ARCHIVE_BYTES", 16)
    streamed = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        streamed["n"] += 1
        return httpx.Response(200, content=b"x" * 4096)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    destination = tmp_path / ASSET_NAME
    with pytest.raises(RuntimeBundleError, match="download_too_large"):
        download_release_asset(destination, client=client)
    assert not destination.exists()


def test_a_malformed_content_length_is_refused(tmp_path: Path) -> None:
    """Pins the exact identity: a bare ``raises`` would pass on any download failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"payload", headers={"content-length": "not-a-number"})

    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("untouched")
    before = snapshot(tmp_path)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    destination = tmp_path / ASSET_NAME
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(destination, client=client)
    assert str(caught.value) == "runtime_bundle_download_length_invalid"
    assert not destination.exists()
    assert snapshot(tmp_path) == before


def test_an_oversized_stream_without_content_length_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that lies about or omits the length must still be bounded."""

    monkeypatch.setattr(runtime_bundle, "MAXIMUM_ARCHIVE_BYTES", 64)

    def stream() -> Any:
        for _ in range(20):
            yield b"x" * 32

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream())

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    destination = tmp_path / ASSET_NAME
    with pytest.raises(RuntimeBundleError, match="download_too_large"):
        download_release_asset(destination, client=client)
    assert not destination.exists(), "a partial download must not be left behind"


def test_an_empty_download_is_refused_and_removed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    destination = tmp_path / ASSET_NAME
    with pytest.raises(RuntimeBundleError, match="download_empty"):
        download_release_asset(destination, client=client)
    assert not destination.exists()


def test_a_failed_download_reports_no_upstream_body(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            content=b"<html>Not Found: secret detail</html>",
            headers={"x-secret-header": "leaked-token"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(tmp_path / ASSET_NAME, client=client)
    message = str(caught.value)
    assert message == "runtime_bundle_download_failed"
    for leak in ("secret", "leaked-token", "html", "token="):
        assert leak not in message


def test_install_from_release_downloads_then_installs(archive: Path, destination: Path) -> None:
    payload = archive.read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == RELEASE_URL
        return httpx.Response(200, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    result = install_from_release(root=destination, client=client)
    assert set(result.installed) == set(BUNDLE_MEMBERS)


def test_the_test_suite_cannot_make_a_real_network_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard: if the mock transport were ever dropped, this must fail rather than dial out."""

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a real network client was constructed")

    monkeypatch.setattr(httpx, "Client", explode)
    with pytest.raises(AssertionError):
        download_release_asset(tmp_path / ASSET_NAME)


def test_the_installation_result_shape_is_stable() -> None:
    result = InstallationResult(installed=("a",), already_present=("b",), evaluation_installed=True)
    assert result.installed == ("a",)
    assert result.already_present == ("b",)
    assert result.evaluation_installed is True


# ============================================================ Gate I0.3: whole-archive pin


def test_the_canonical_archive_matches_the_pinned_digest(archive: Path) -> None:
    assert digest(archive.read_bytes()) == FIXTURE_ARCHIVE_SHA256


def test_the_real_pinned_archive_digest_is_a_source_constant() -> None:
    """Outer provenance lives in source, never inside the container it judges."""

    assert len(REAL_EXPECTED_ARCHIVE_SHA256) == 64
    assert REAL_EXPECTED_ARCHIVE_SHA256 == (
        "5f789aecdd74ab31a92cfdb9da5d8d1312e89ac488b79a763374b5e425046cfe"
    )


def test_the_archive_never_declares_its_own_outer_hash(archive: Path) -> None:
    """A container that states the value it is judged by proves nothing."""

    rendered = members_of(archive)[BUNDLE_MANIFEST_NAME].decode()
    assert FIXTURE_ARCHIVE_SHA256 not in rendered
    assert REAL_EXPECTED_ARCHIVE_SHA256 not in rendered
    assert "archive_sha256" not in rendered


def test_a_single_flipped_byte_is_rejected_at_the_archive_boundary(archive: Path) -> None:
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    archive.write_bytes(bytes(raw))
    with pytest.raises(RuntimeBundleError) as caught:
        read_validated_archive(archive)
    assert str(caught.value) == "runtime_bundle_archive_hash_mismatch"


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("appended", lambda raw: raw + b"trailing"),
        ("prepended", lambda raw: b"MZ-self-extracting-stub" + raw),
        ("truncated", lambda raw: raw[:-40]),
        ("eocd-rewritten", lambda raw: raw[:-22] + b"\x00" * 22),
    ],
)
def test_structurally_malicious_bytes_fail_at_the_archive_hash_boundary(
    archive: Path, label: str, mutate: Any
) -> None:
    """Every reshaping stops at the outer pin, before any format arithmetic runs."""

    archive.write_bytes(mutate(archive.read_bytes()))
    with pytest.raises(RuntimeBundleError) as caught:
        read_validated_archive(archive)
    assert str(caught.value) == "runtime_bundle_archive_hash_mismatch", label


def test_eocd_parsing_is_not_invoked_after_an_archive_hash_mismatch(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(runtime_bundle, "raw_member_names", lambda payload: calls.append(1) or [])
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(RuntimeBundleError, match="archive_hash_mismatch"):
        read_validated_archive(archive)
    assert calls == [], "the EOCD parser ran on unverified bytes"


def test_zipfile_is_not_constructed_after_an_archive_hash_mismatch(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("zipfile.ZipFile was constructed on unverified bytes")

    monkeypatch.setattr(runtime_bundle.zipfile, "ZipFile", explode)
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(RuntimeBundleError, match="archive_hash_mismatch"):
        read_validated_archive(archive)


def test_no_destination_mutation_on_an_archive_hash_mismatch(
    archive: Path, destination: Path
) -> None:
    keep = destination / "notes.txt"
    keep.write_text("untouched")
    before = snapshot(destination)

    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(RuntimeBundleError, match="archive_hash_mismatch"):
        install_from_archive(archive, root=destination)
    assert snapshot(destination) == before


def test_the_installer_and_downloader_both_pass_through_the_archive_pin(
    archive: Path, destination: Path
) -> None:
    """The download path reaches installation through the same validator, not around it."""

    payload = archive.read_bytes() + b"x"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError, match="archive_hash_mismatch"):
        install_from_release(root=destination, client=client)
    assert snapshot(destination) == {}


# ============================================================ Gate I0.3: transport-error cleanup


def test_a_mid_stream_transport_error_removes_the_partial_download(tmp_path: Path) -> None:
    """``httpx.HTTPError`` is not an ``OSError``, so the old cleanup never ran for it.

    At least one chunk is delivered before the failure, so the destination genuinely exists and is
    genuinely partial at the moment the error is raised.
    """

    def chunks() -> Any:
        yield b"partial-archive-bytes"
        raise httpx.ReadError("connection dropped mid-stream")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks())

    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("untouched")
    destination = tmp_path / ASSET_NAME

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(destination, client=client)

    assert str(caught.value) == "runtime_bundle_download_failed"
    assert not destination.exists(), "a partial download was left behind"
    assert unrelated.read_text() == "untouched"
    for leak in ("connection dropped", "partial-archive-bytes", "http"):
        assert leak not in str(caught.value)


def test_a_download_never_touches_a_pre_existing_file(tmp_path: Path) -> None:
    """The test that used to pass on a destroyed file.

    It asserted only ``destination.exists()``. Since ``open("wb")`` had already truncated the file
    to zero bytes, an emptied file satisfied it -- the guarantee in the name was never checked. The
    content assertion below is the whole point, and it fails against the old implementation.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    destination = tmp_path / ASSET_NAME
    destination.write_bytes(b"someone elses file")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(destination, client=client)

    assert str(caught.value) == "runtime_bundle_download_destination_exists"
    assert destination.read_bytes() == b"someone elses file"


def test_an_existing_empty_destination_is_also_preserved(tmp_path: Path) -> None:
    """Zero bytes is still somebody's file, and still not ours to take."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"payload")

    destination = tmp_path / ASSET_NAME
    destination.write_bytes(b"")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(destination, client=client)

    assert str(caught.value) == "runtime_bundle_download_destination_exists"
    assert destination.exists()
    assert destination.read_bytes() == b""


def test_a_successful_response_cannot_replace_an_existing_destination(tmp_path: Path) -> None:
    """A 200 with real content is exactly the case that used to overwrite."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"a perfectly valid archive would go here")

    destination = tmp_path / ASSET_NAME
    destination.write_bytes(b"original contents")
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("unrelated")
    before = snapshot(tmp_path)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError) as caught:
        download_release_asset(destination, client=client)

    assert str(caught.value) == "runtime_bundle_download_destination_exists"
    assert snapshot(tmp_path) == before, "no byte anywhere may change"
    assert sibling.read_text() == "unrelated"


def test_a_failed_response_cannot_replace_or_delete_an_existing_destination(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"upstream exploded")

    destination = tmp_path / ASSET_NAME
    destination.write_bytes(b"original contents")
    before = snapshot(tmp_path)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError):
        download_release_asset(destination, client=client)

    assert destination.exists()
    assert snapshot(tmp_path) == before


def test_no_payload_is_written_when_exclusive_acquisition_fails(tmp_path: Path) -> None:
    """Structurally observable: the response body is large and distinctive, and never lands."""

    marker = b"X" * 4096

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=marker)

    destination = tmp_path / ASSET_NAME
    destination.write_bytes(b"tiny")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(RuntimeBundleError, match="download_destination_exists"):
        download_release_asset(destination, client=client)

    payload = destination.read_bytes()
    assert payload == b"tiny"
    assert marker not in payload
    assert len(payload) == 4


def test_ownership_is_not_inferred_from_a_stale_existence_check() -> None:
    """The defect class, asserted against the source rather than the behaviour.

    A truncating open guarded by an earlier ``exists()`` is the shape being excluded; asserting it
    is gone stops it being reintroduced by a well-meaning refactor.
    """

    source = Path(runtime_bundle.__file__).read_text(encoding="utf-8")
    assert "O_EXCL" in source, "exclusive creation is the ownership mechanism"
    assert "destination.exists()" not in source
    assert 'destination.open("wb")' not in source


# ------------------------------------------------------------------ F1: bounded archive read


class _CountingHandle:
    """A file handle that reports one size and can deliver another."""

    def __init__(self, payload: bytes, limits: list[int | None]) -> None:
        self._payload = payload
        self._limits = limits

    def read(self, limit: int | None = None) -> bytes:
        self._limits.append(limit)
        return self._payload if limit is None else self._payload[:limit]

    def fileno(self) -> int:
        return 4242

    def __enter__(self) -> "_CountingHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _serve_bytes(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, reported_size: int
) -> list[int | None]:
    """Make one archive path report ``reported_size`` while delivering ``payload``."""

    limits: list[int | None] = []
    real_open = Path.open

    def fake_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == ASSET_NAME:
            return _CountingHandle(payload, limits)
        return real_open(self, *args, **kwargs)

    class _Stat:
        st_size = reported_size

    real_fstat = runtime_bundle.os.fstat
    monkeypatch.setattr(Path, "open", fake_open)
    monkeypatch.setattr(
        runtime_bundle.os,
        "fstat",
        lambda fd: _Stat() if fd == 4242 else real_fstat(fd),
    )
    return limits


def test_the_archive_read_is_always_bounded(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The read must carry an explicit limit, never be an unbounded ``read()``."""

    payload = archive.read_bytes()
    limits = _serve_bytes(monkeypatch, payload, len(payload))
    read_validated_archive(archive)

    assert limits == [runtime_bundle.MAXIMUM_ARCHIVE_BYTES + 1]
    assert None not in limits, "an unbounded read was issued"


def test_a_file_that_grows_after_its_reported_size_is_rejected(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check-then-act race: a small reported size, then more bytes than the ceiling.

    The bound has to come from what was actually delivered, not from the earlier ``fstat``.
    """

    monkeypatch.setattr(runtime_bundle, "MAXIMUM_ARCHIVE_BYTES", 64)
    oversized = b"Z" * 4096
    _serve_bytes(monkeypatch, oversized, reported_size=16)

    with pytest.raises(RuntimeBundleError) as caught:
        read_validated_archive(archive)
    assert str(caught.value) == "runtime_bundle_archive_too_large"


def test_no_parser_runs_after_an_oversized_archive_is_rejected(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("zipfile.ZipFile was constructed on an oversized archive")

    monkeypatch.setattr(runtime_bundle, "MAXIMUM_ARCHIVE_BYTES", 64)
    monkeypatch.setattr(runtime_bundle, "raw_member_names", lambda payload: calls.append(1) or [])
    monkeypatch.setattr(runtime_bundle.zipfile, "ZipFile", explode)
    _serve_bytes(monkeypatch, b"Z" * 4096, reported_size=16)

    with pytest.raises(RuntimeBundleError, match="archive_too_large"):
        read_validated_archive(archive)
    assert calls == [], "the EOCD parser ran on an oversized archive"


def test_an_oversized_archive_mutates_no_destination(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keep = destination / "notes.txt"
    keep.write_text("untouched")
    before = snapshot(destination)

    monkeypatch.setattr(runtime_bundle, "MAXIMUM_ARCHIVE_BYTES", 64)
    _serve_bytes(monkeypatch, b"Z" * 4096, reported_size=16)

    with pytest.raises(RuntimeBundleError, match="archive_too_large"):
        install_from_archive(archive, root=destination)
    assert snapshot(destination) == before


def test_an_empty_archive_is_still_refused_by_the_bounded_read(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_bytes(monkeypatch, b"", reported_size=0)
    with pytest.raises(RuntimeBundleError) as caught:
        read_validated_archive(archive)
    assert str(caught.value) == "runtime_bundle_archive_empty"


def test_the_bounded_read_leaves_canonical_validation_unchanged(archive: Path) -> None:
    """The real path, with no patching at all."""

    assert set(read_validated_archive(archive)) == set(BUNDLE_MEMBERS)


# ============================================================ Gate I0.3: relative-path symlinks


def test_a_relative_archive_path_is_inspected_at_its_real_location(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The misjoin bug: a relative path was joined to the boundary, not the working directory.

    ``dist/bundle.zip`` became ``C:\\dist\\bundle.zip`` -- a path that does not exist, so the
    symlink check inspected nothing and always passed.
    """

    monkeypatch.chdir(archive.parent)
    relative = Path(archive.name)
    assert (
        digest(read_validated_archive(relative)[EVALUATION_MEMBER])
        == (FIXTURE_HASHES[EVALUATION_MEMBER])
    )


def test_an_absolute_archive_path_still_works(archive: Path) -> None:
    assert set(read_validated_archive(archive)) == set(BUNDLE_MEMBERS)


def test_the_absolutiser_uses_the_working_directory_not_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pure path arithmetic, so it holds on every platform including Windows drive roots."""

    monkeypatch.chdir(tmp_path)
    assert runtime_bundle._absolute(Path("dist/x.zip")) == tmp_path / "dist" / "x.zip"
    absolute = tmp_path / "already" / "absolute.zip"
    assert runtime_bundle._absolute(absolute) == absolute
    # A drive root or UNC anchor is preserved rather than rewritten.
    anchor = Path(absolute.anchor)
    assert runtime_bundle._absolute(anchor) == anchor


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="platform cannot create symbolic links")
def test_a_symlinked_relative_archive_path_is_refused(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipped rather than silently passing where symlink creation is unavailable."""

    link = tmp_path / "link.zip"
    try:
        link.symlink_to(archive)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable to this process")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeBundleError) as caught:
        read_validated_archive(Path("link.zip"))
    assert str(caught.value) == "runtime_bundle_symlinked_path"


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="platform cannot create symbolic links")
def test_a_symlinked_parent_of_a_relative_archive_path_is_refused(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked_dir = tmp_path / "viadir"
    try:
        linked_dir.symlink_to(archive.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable to this process")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeBundleError) as caught:
        read_validated_archive(Path("viadir") / archive.name)
    assert str(caught.value) == "runtime_bundle_symlinked_parent"


def test_a_relative_install_root_is_absolutised(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper contract is absolute; entry points make it true rather than assuming it."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / "clone").mkdir()
    result = install_from_archive(archive, root=Path("clone"))
    assert set(result.installed) == set(BUNDLE_MEMBERS)
    for member in BUNDLE_MEMBERS:
        assert (tmp_path / "clone" / member).read_bytes() == FIXTURES[member]


# ============================================================ Gate I0.3: structural ZIP64


def test_a_structurally_positioned_zip64_locator_is_refused(archive: Path) -> None:
    """The locator has exactly one legal position: the 20 bytes before the EOCD."""

    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    raw[eocd:eocd] = b"PK\x06\x07" + b"\x00" * 16
    refuses(rebuild_raw(archive, bytes(raw)), "runtime_bundle_archive_zip64_unsupported")


@pytest.mark.parametrize(
    ("offset", "width", "label"),
    [
        (4, 2, "disk number"),
        (6, 2, "directory disk"),
        (8, 2, "records on disk"),
        (10, 2, "total records"),
        (12, 4, "directory size"),
        (16, 4, "directory offset"),
    ],
)
def test_a_zip64_sentinel_field_is_refused(
    archive: Path, offset: int, width: int, label: str
) -> None:
    """A sentinel means the real value lives in a ZIP64 record; refused, never truncated."""

    raw = bytearray(archive.read_bytes())
    eocd = len(raw) - 22
    raw[eocd + offset : eocd + offset + width] = b"\xff" * width
    refuses(rebuild_raw(archive, bytes(raw)), "runtime_bundle_archive_zip64_unsupported")


def test_zip64_like_bytes_inside_member_data_do_not_classify_the_archive(
    tmp_path: Path,
) -> None:
    """Member content is data, not format. The old whole-payload scan let it decide otherwise."""

    hostile = tmp_path / "hostile.zip"
    with zipfile.ZipFile(hostile, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("a.json", b"PK\x06\x06" + b"\x00" * 40 + b"PK\x06\x07")
        handle.writestr("b.json", b"plain")

    payload = hostile.read_bytes()
    assert b"PK\x06\x06" in payload and b"PK\x06\x07" in payload
    # Parses cleanly: the decoy bytes never reach the format classification.
    assert raw_member_names(payload) == ["a.json", "b.json"]


def test_the_canonical_archive_still_validates_unchanged(archive: Path) -> None:
    assert set(read_validated_archive(archive)) == set(BUNDLE_MEMBERS)


# ============================================================ Gate I0.3: concurrency boundary


def test_two_installers_publishing_the_same_bundle_cannot_corrupt_content(
    archive: Path, destination: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic stand-in for a race, with no threads and no sleeps.

    A second installer is simulated by writing the same pinned bytes to a destination immediately
    after this installer classified it as absent -- the exact interval the module documents as
    outside its trust boundary. Because only one byte sequence can ever be published, the outcome
    is still correct content, which is the guarantee actually claimed.
    """

    real_classify = runtime_bundle._classify_destination
    raced: list[str] = []

    def classify_then_race(path: Path, payload: bytes, root: Path) -> bool:
        result = real_classify(path, payload, root)
        if not result and not raced:
            raced.append(path.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)  # the "other" installer wins the race
        return result

    monkeypatch.setattr(runtime_bundle, "_classify_destination", classify_then_race)
    install_from_archive(archive, root=destination)

    assert raced, "the race was never exercised"
    for member in BUNDLE_MEMBERS:
        assert (destination / member).read_bytes() == FIXTURES[member]
    verify_installation(root=destination, require_evaluation=True)
