"""Deterministic, hash-pinned distribution of the locked runtime artifacts.

A clean ``git clone`` cannot start RiskLoom. Every artifact the service binds to at startup is
Git-ignored, deliberately: committing generated artifacts would misrepresent them as source, and
regenerating them elsewhere would produce a *different* model and break every hash the project
publishes. Until now the only documented remedy was "copy the supplied bundle into place", which is
not a workflow a stranger can follow.

This module closes that gap without weakening anything. It packages exactly five canonical JSON
artifacts into a byte-deterministic ZIP, pins their SHA-256 values in source, and installs them
only after the whole archive has been validated. Every path, every hash and the single download URL
are compile-time constants: there is no caller-supplied URL, no environment override, no discovery
step, and no general downloader hiding in here.

Scope, deliberately narrow. This module moves already-approved bytes. It does not train, evaluate,
score, generate features, read labels, reopen the protected held-out partition, or contact Razorpay,
Gemini or PostgreSQL. The one thing it does *after* installing is call the ordinary
``load_serving_bundle()`` the application itself uses, so a bundle that installs cleanly but cannot
serve is reported as a failure rather than a success.

Two independent layers of provenance, and both must hold. The whole archive is pinned by
``EXPECTED_ARCHIVE_SHA256`` and checked before any parsing at all, so hostile bytes meet a
64-character comparison rather than format arithmetic; the five members are pinned separately and
remain authoritative on their own. The archive never records its own outer hash -- a container that
declares the value it is judged by proves nothing.

Every error is a short stable identity. Artifact contents are never printed or logged, and neither
is any upstream response body, header, query string or signed URL. A download that fails for any
reason removes the file that attempt created, and never one that was already there.

What this module does **not** claim: it is not a hardened general-purpose ZIP reader. It accepts
exactly the narrow archive shape it produces itself and rejects everything else explicitly, which
is a much smaller thing to get right than tolerating the format's full variety.

Nor is publication a transaction. Files are classified first and published second, and the two are
not one atomic step across three directories. The trust boundary is the local host: paths under
``artifacts/`` are assumed not to be rewritten by another authorised local process during the short
interval between classification and publication. That assumption is not a weakness worth engineering
around, because a process able to win that race is equally able to rewrite the artifacts a moment
after installation finishes. What *is* guaranteed is that two RiskLoom installers can only ever
publish the same pinned byte sequences, so concurrent installs cannot interleave into corrupt
content, and that an installer never deliberately overwrites content it observed to differ.
"""

import contextlib
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import httpx

from riskloom.modeling.artifacts import MODEL_ARTIFACT_FILENAMES
from riskloom.modeling.canonical import canonical_json_bytes
from riskloom.serving.model_host import ServingBundleError, load_serving_bundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PRODUCT = "RiskLoom"
ARTIFACT_TYPE = "runtime_artifact_bundle"
BUNDLE_SCHEMA_VERSION = "1.0.0"

# Two different tags, deliberately.
#
# ``RELEASE_TAG`` names the immutable runtime-artifact release the installer downloads from. It is
# pinned rather than discovered: an installer that resolved "latest" would silently change what a
# clean clone receives. It must stay fixed once published, because a published bundle's own manifest
# records it.
#
# ``SUBMISSION_TAG`` names the source snapshot and has nothing to do with the download. Keeping them
# separate means the source can be re-tagged -- for a documentation fix, say -- without invalidating
# a published runtime asset or forcing a re-upload. Nothing in this module may assume they match.
RELEASE_TAG = "v1.0.3-runtime"
SUBMISSION_TAG = "v1.0.3-submission"

ASSET_NAME = "riskloom-runtime-artifacts.zip"
RELEASE_URL = (
    f"https://github.com/XxSURYANSHxX/riskloom/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
)

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"

MODEL_DIRECTORY = "artifacts/models/development"
FEATURE_DIRECTORY = "artifacts/features/development-v1.1.0-config-bound"
EVALUATION_DIRECTORY = "artifacts/evaluations/development"

# Required for the service to bind and serve at all.
STARTUP_MEMBERS = (
    f"{MODEL_DIRECTORY}/model.json",
    f"{MODEL_DIRECTORY}/training_report.json",
    f"{MODEL_DIRECTORY}/manifest.json",
    f"{FEATURE_DIRECTORY}/manifest.json",
)

# Required for the complete submitted experience: the dashboard's model panel, the drift reference
# and the published held-out aggregates. Absent, the service still starts and those surfaces report
# an ordinary unavailable state.
EVALUATION_MEMBER = f"{EVALUATION_DIRECTORY}/evaluation.json"

BUNDLE_MEMBERS = (*STARTUP_MEMBERS, EVALUATION_MEMBER)

# Pinned in source and immutable. These are the identities every other gate in this project
# publishes; an installer that trusted the archive's own declaration alone would accept a bundle
# whose manifest had simply been re-signed around a tampered payload.
EXPECTED_SHA256: Any = MappingProxyType(
    {
        f"{MODEL_DIRECTORY}/model.json": (
            "3db8dafef643261c0df559cab632cfaf6fc45be54f38c1f4a621ef5af84039d4"
        ),
        f"{MODEL_DIRECTORY}/training_report.json": (
            "4ff96556f1df49d7c44c29703c328753a6ea0ef197410976100e84751943ea6d"
        ),
        f"{MODEL_DIRECTORY}/manifest.json": (
            "00aa16380eee1dcfa26fe9c89ed0eb8f866e75e98bd7a7ba89f9cc228c792f2e"
        ),
        f"{FEATURE_DIRECTORY}/manifest.json": (
            "15337d0e9220f7ca96b4ded8157bc3ad29f38a6c5db9d357dea00b09371f28ba"
        ),
        EVALUATION_MEMBER: "11251cef0dade5d14d2d1a85fe3822126e01c2a354494dd09b720c679244c40d",
    }
)

# The whole published archive, pinned in source. This is deliberately *outer* provenance: it is not
# recorded anywhere inside the archive, because a container that declares its own authoritative hash
# is self-referential and proves nothing. Checked before a single byte reaches the EOCD parser,
# the central-directory parser, ``zipfile``, or the manifest reader, so hostile bytes meet a
# 64-character comparison rather than hand-written format arithmetic.
#
# It does not replace the five member pins below and cannot be traded against them. The members stay
# independently authoritative, so a future re-cut that changed the container without changing the
# artifacts would still have to satisfy both.
EXPECTED_ARCHIVE_SHA256 = "5f789aecdd74ab31a92cfdb9da5d8d1312e89ac488b79a763374b5e425046cfe"

# One canonical allowlist, enforced at import. Three collections describe the member set and they
# must never drift apart; asserting it here means a future edit that adds a member to one and
# forgets the others fails immediately rather than shipping an installer that cannot verify what it
# installs.
assert set(EXPECTED_SHA256) == set(BUNDLE_MEMBERS), "pinned hashes and member list disagree"
assert set(STARTUP_MEMBERS) | {EVALUATION_MEMBER} == set(BUNDLE_MEMBERS)
assert len(BUNDLE_MEMBERS) == len(set(BUNDLE_MEMBERS)), "duplicate member"
assert BUNDLE_MANIFEST_NAME not in BUNDLE_MEMBERS

ARCHIVE_NAMES = frozenset({*BUNDLE_MEMBERS, BUNDLE_MANIFEST_NAME})

# Directories whose contents are validated byte-exactly elsewhere. Nothing this module writes may
# land inside one except an approved member: an unexpected file in the model directory makes
# ``load_locked_model`` fail with ``locked_model_artifact_set_invalid``.
PROTECTED_DIRECTORIES = (MODEL_DIRECTORY, FEATURE_DIRECTORY, EVALUATION_DIRECTORY)

# The largest approved artifact is roughly 29 KB. These ceilings are generous enough to survive an
# ordinary re-lock and small enough that a decompression bomb cannot reach the filesystem.
MAXIMUM_MEMBER_BYTES = 8 * 1024 * 1024
MAXIMUM_TOTAL_BYTES = 32 * 1024 * 1024
MAXIMUM_ARCHIVE_BYTES = 16 * 1024 * 1024

DOWNLOAD_TIMEOUT_SECONDS = 60.0
MAXIMUM_REDIRECTS = 5
PERMITTED_DOWNLOAD_PORTS = frozenset({None, 443})

# Exact hosts only. No suffix matching: ``*.githubusercontent.com`` would admit any user-controlled
# subdomain, and the release redirect chain needs only these three.
PERMITTED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)

# Fixed ZIP metadata. The DOS epoch removes wall-clock time; 0o644 removes the build machine's
# umask; ZIP_STORED removes any dependence on a zlib version's deflate output. Together these make
# two builds from the same approved bytes byte-identical.
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o644
ZIP_COMPRESSION = zipfile.ZIP_STORED
ZIP_CREATE_SYSTEM = 3  # Unix, so the mode above is meaningful and not host-dependent.

STAGING_PREFIX = ".riskloom-bundle-staging-"
DOWNLOAD_PREFIX = "riskloom-bundle-download-"


class RuntimeBundleError(RuntimeError):
    """A safe bundle error. ``str()`` is a stable identity and never carries file content."""


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Byte size and digest of one approved artifact."""

    byte_size: int
    sha256: str

    def as_json(self) -> dict[str, Any]:
        return {"byte_size": self.byte_size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class InstallationResult:
    """What an installation actually did, for the CLI to report."""

    installed: tuple[str, ...]
    already_present: tuple[str, ...]
    evaluation_installed: bool


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: bytes) -> dict[str, Any]:
    """Parse strictly canonical JSON, or refuse.

    Mirrors ``riskloom.modeling.canonical.read_canonical_json`` but operates on bytes already in
    memory, because archive members are validated before anything touches the filesystem.
    """

    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeBundleError("runtime_bundle_json_invalid") from None
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise RuntimeBundleError("runtime_bundle_json_not_canonical")
    return value


def _absolute(path: Path) -> Path:
    """Absolutise lexically, against the working directory, without resolving symlinks.

    ``resolve()`` is deliberately avoided: it follows links, which would hide the very link the
    caller is about to reject. Only ``is_symlink`` (an ``lstat``) inspects the filesystem, and it
    never traverses.
    """

    return path if path.is_absolute() else (Path.cwd() / path)


def _refuse_symlinked(path: Path, *, boundary: Path | None = None) -> None:
    """Refuse a path that is, or sits beneath, a symbolic link.

    Checked on both sides: a symlinked source would package something other than the approved file,
    and a symlinked destination would write through the link to a location outside the artifact
    tree. ``boundary`` stops the upward walk at the tree being operated on, so a temporary
    destination root is not penalised for whatever sits above it; without one the walk runs to the
    filesystem anchor, which is correct for a path supplied from outside any known tree.

    A relative path is absolutised against the working directory, not against the boundary. Joining
    it to the boundary produced a path that did not exist -- ``C:\\dist\\bundle.zip`` for
    ``dist/bundle.zip`` -- so the check silently inspected the wrong location and always passed.
    """

    candidate = _absolute(path)
    if candidate.is_symlink():
        raise RuntimeBundleError("runtime_bundle_symlinked_path")

    stop = _absolute(boundary) if boundary is not None else None
    for parent in candidate.parents:
        if parent.is_symlink():
            raise RuntimeBundleError("runtime_bundle_symlinked_parent")
        # ``parent == parent.parent`` is the filesystem anchor, and preserves Windows drive roots
        # and UNC shares, where the anchor is reached without ever equalling "/".
        if parent == stop or parent == parent.parent:
            break


# --------------------------------------------------------------------------------- build


def _read_approved_artifact(root: Path, member: str) -> bytes:
    """Read one approved artifact from disk with every source check applied."""

    path = root / member
    _refuse_symlinked(path, boundary=root)
    if not path.exists():
        raise RuntimeBundleError("runtime_bundle_source_missing")
    if not path.is_file():
        raise RuntimeBundleError("runtime_bundle_source_not_a_file")
    try:
        payload = path.read_bytes()
    except OSError:
        raise RuntimeBundleError("runtime_bundle_source_unreadable") from None
    if not payload:
        raise RuntimeBundleError("runtime_bundle_source_empty")
    if len(payload) > MAXIMUM_MEMBER_BYTES:
        raise RuntimeBundleError("runtime_bundle_source_too_large")
    # Canonical JSON is the project-wide artifact contract; a substituted file that merely hashes
    # differently is caught below, but one that is not canonical JSON is not an artifact at all.
    _canonical_json(payload)
    if _sha256(payload) != EXPECTED_SHA256[member]:
        raise RuntimeBundleError("runtime_bundle_source_hash_mismatch")
    return payload


def build_manifest(artifacts: dict[str, bytes]) -> dict[str, Any]:
    """The bundle's own manifest.

    It records what the archive claims to be and what it claims to carry: product marker, artifact
    type, schema version, the runtime release tag, the member allowlist, and each member's exact
    size and digest, plus the source model and feature-dataset identities.

    It never records its own hash, matching every other manifest in this repository, and it carries
    no hostname, path, timestamp, locale, username or machine data -- every value is either a
    compile-time constant or derived from the approved bytes.
    """

    model_manifest = _canonical_json(artifacts[f"{MODEL_DIRECTORY}/manifest.json"])
    feature_manifest = _canonical_json(artifacts[f"{FEATURE_DIRECTORY}/manifest.json"])
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifacts": {
            member: ArtifactMetadata(len(payload), _sha256(payload)).as_json()
            for member, payload in sorted(artifacts.items())
        },
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "members": sorted(artifacts),
        "product": PRODUCT,
        "release_tag": RELEASE_TAG,
        "source": {
            "feature_dataset_id": feature_manifest["feature_dataset_id"],
            "model_id": model_manifest["model_id"],
        },
    }


def _refuse_unsafe_output(output: Path, root: Path) -> Path:
    """Refuse an output path that is a well-known directory, protected, symlinked, or occupied.

    The protected-directory rule is not theoretical. Writing the archive into
    ``artifacts/models/development/`` leaves a fourth file beside the three the strict loader
    permits, so the very model directory this bundle exists to deliver would stop loading.
    """

    try:
        resolved = output.resolve(strict=False)
    except (OSError, RuntimeError):
        raise RuntimeBundleError("runtime_bundle_output_unsafe") from None

    forbidden = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        REPOSITORY_ROOT,
        root.resolve(),
        Path(resolved.anchor).resolve(),
    }
    if resolved in forbidden or resolved.parent == resolved:
        raise RuntimeBundleError("runtime_bundle_output_unsafe")

    for protected in PROTECTED_DIRECTORIES:
        directory = (root / protected).resolve()
        if resolved == directory or directory in resolved.parents:
            raise RuntimeBundleError("runtime_bundle_output_inside_artifact_tree")

    # A source artifact is never a legal output, even under a different spelling of the same path.
    for member in BUNDLE_MEMBERS:
        if resolved == (root / member).resolve():
            raise RuntimeBundleError("runtime_bundle_output_is_source_artifact")

    if output.is_symlink():
        raise RuntimeBundleError("runtime_bundle_symlinked_path")
    if output.exists():
        raise RuntimeBundleError("runtime_bundle_output_exists")
    _refuse_symlinked(resolved.parent, boundary=Path(resolved.anchor))
    return resolved


def build_bundle(output: Path, *, root: Path | None = None) -> Path:
    """Package the approved local artifacts into a deterministic archive.

    The source bytes are never modified. Two builds from the same approved inputs produce
    byte-identical archives, so the published asset can be reproduced and compared.
    """

    # Absolute from here on. Every path helper below assumes it, and a relative root silently
    # produced doubled prefixes that inspected the wrong location.
    source_root = REPOSITORY_ROOT if root is None else _absolute(root)
    resolved = _refuse_unsafe_output(output, source_root)

    artifacts = {member: _read_approved_artifact(source_root, member) for member in BUNDLE_MEMBERS}
    manifest_bytes = canonical_json_bytes(build_manifest(artifacts))

    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Written to a sibling temporary name and moved into place, so an interrupted build never
    # leaves a half-written archive that looks publishable.
    staging = resolved.parent / f".{resolved.name}.partial"
    try:
        with zipfile.ZipFile(staging, "w", compression=ZIP_COMPRESSION) as archive:
            # Fixed order: approved artifacts sorted, then the manifest last, mirroring the
            # manifest-last publication convention used by every other artifact writer here.
            for member in sorted(artifacts):
                _write_member(archive, member, artifacts[member])
            _write_member(archive, BUNDLE_MANIFEST_NAME, manifest_bytes)

        # The build proves it reproduced the pinned container before anything is published. A
        # deterministic builder that has silently started emitting different bytes is exactly the
        # case a downstream installer could not distinguish from tampering, so it is caught here,
        # while the output is still a temporary file that no one has been handed.
        if _sha256(staging.read_bytes()) != EXPECTED_ARCHIVE_SHA256:
            raise RuntimeBundleError("runtime_bundle_build_hash_mismatch")

        os.replace(staging, resolved)
    except OSError:
        raise RuntimeBundleError("runtime_bundle_build_failed") from None
    finally:
        # Runs on every exit, so a mismatching or interrupted build leaves no partial archive and
        # never replaces whatever was already at the destination.
        if staging.exists():
            staging.unlink(missing_ok=True)
    return resolved


def _write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    """Write one member with fully fixed metadata and no comment or extra field."""

    info = zipfile.ZipInfo(filename=name, date_time=ZIP_TIMESTAMP)
    info.compress_type = ZIP_COMPRESSION
    info.create_system = ZIP_CREATE_SYSTEM
    info.external_attr = (ZIP_FILE_MODE & 0xFFFF) << 16
    info.comment = b""
    info.extra = b""
    archive.writestr(info, payload)


# --------------------------------------------------------------- raw central-directory parsing

CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
CENTRAL_DIRECTORY_FIXED_BYTES = 46
END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"
END_OF_CENTRAL_DIRECTORY_FIXED_BYTES = 22
# Only the locator is named, because only the locator is checked. The ZIP64 end-of-central-directory
# record is reachable solely through this locator, and any EOCD field that would require it carries
# a sentinel that is rejected below, so a second constant would name a check that does not exist.
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_LOCATOR_BYTES = 20  # Fixed size; sits immediately before the EOCD when present.
U16_SENTINEL = 0xFFFF  # "the real value is in a ZIP64 record"
U32_SENTINEL = 0xFFFFFFFF
REQUIRED_EOCD_COMMENT_BYTES = 0  # This bundle never writes an archive comment.
MAXIMUM_CENTRAL_DIRECTORY_RECORDS = 64
UTF8_NAME_FLAG = 0x800


def _u16(payload: bytes, offset: int) -> int:
    return int.from_bytes(payload[offset : offset + 2], "little")


def _u32(payload: bytes, offset: int) -> int:
    return int.from_bytes(payload[offset : offset + 4], "little")


def raw_member_names(payload: bytes) -> list[str]:
    """Member names exactly as the archive stores them, before ``zipfile`` normalises anything.

    This exists because ``ZipFile`` rewrites ``\\`` to ``/`` while reading the central directory, so
    a backslash check applied to ``ZipInfo.filename`` can never fire. Such an archive would still be
    refused as an unknown member, but "rejected for the right reason" and "rejected by accident" are
    different guarantees, and only the first survives a future change to the allowlist.

    The parse is anchored at the end-of-central-directory record rather than scanning for record
    signatures. Scanning is unsound: ``PK\\x01\\x02`` occurs freely inside stored member data, so a
    scanning parser reads attacker-chosen bytes as though they were directory records. Every field
    read here is bounds-checked first, the record count and directory extent are validated against
    the archive size, and ZIP64 and multi-disk archives are refused rather than partially handled.
    """

    size = len(payload)
    if size < END_OF_CENTRAL_DIRECTORY_FIXED_BYTES:
        raise RuntimeBundleError("runtime_bundle_archive_truncated")

    # The EOCD is the last record. With no comment it sits at exactly the final 22 bytes, which is
    # the only shape this bundle writes; a bounded search over the legal trailing window is still
    # performed so that an archive carrying a comment is rejected as *a comment* rather than as
    # unrecognisable trailing bytes. The window is the 22-byte record plus the largest comment the
    # format permits, so the scan can never run over the whole file.
    eocd = size - END_OF_CENTRAL_DIRECTORY_FIXED_BYTES
    if payload[eocd : eocd + 4] != END_OF_CENTRAL_DIRECTORY_SIGNATURE:
        window_start = max(0, size - END_OF_CENTRAL_DIRECTORY_FIXED_BYTES - 0xFFFF)
        found = payload.rfind(END_OF_CENTRAL_DIRECTORY_SIGNATURE, window_start)
        if found == -1:
            raise RuntimeBundleError("runtime_bundle_archive_eocd_invalid")
        if found + END_OF_CENTRAL_DIRECTORY_FIXED_BYTES > size:
            raise RuntimeBundleError("runtime_bundle_archive_truncated")
        declared = _u16(payload, found + 20)
        if found + END_OF_CENTRAL_DIRECTORY_FIXED_BYTES + declared != size:
            raise RuntimeBundleError("runtime_bundle_archive_eocd_invalid")
        # A well-formed EOCD sitting earlier than the end means the archive carries a comment.
        raise RuntimeBundleError("runtime_bundle_archive_comment_unsupported")

    comment_length = _u16(payload, eocd + 20)
    if comment_length != REQUIRED_EOCD_COMMENT_BYTES:
        raise RuntimeBundleError("runtime_bundle_archive_comment_unsupported")

    # ZIP64 is detected structurally, never by scanning the payload. The locator has exactly one
    # legal position -- the twenty bytes immediately preceding the EOCD -- so that is the only place
    # examined. Scanning the whole file for the signature was the same unsound technique this parser
    # exists to avoid: ``PK\x06\x06`` inside a member's data is ordinary content, not a format
    # marker, and letting it classify the archive hands that decision to the attacker.
    locator = eocd - ZIP64_LOCATOR_BYTES
    if locator >= 0 and payload[locator : locator + 4] == ZIP64_LOCATOR_SIGNATURE:
        raise RuntimeBundleError("runtime_bundle_archive_zip64_unsupported")

    disk_number = _u16(payload, eocd + 4)
    directory_disk = _u16(payload, eocd + 6)
    records_on_disk = _u16(payload, eocd + 8)
    total_records = _u16(payload, eocd + 10)
    directory_size = _u32(payload, eocd + 12)
    directory_offset = _u32(payload, eocd + 16)

    # A sentinel in any of these fields means the real value lives in a ZIP64 record. Refused rather
    # than followed, so a truncated 32-bit reading of a 64-bit value can never be acted on.
    if (
        disk_number == U16_SENTINEL
        or directory_disk == U16_SENTINEL
        or records_on_disk == U16_SENTINEL
        or total_records == U16_SENTINEL
        or directory_size == U32_SENTINEL
        or directory_offset == U32_SENTINEL
    ):
        raise RuntimeBundleError("runtime_bundle_archive_zip64_unsupported")

    if disk_number != 0 or directory_disk != 0:
        raise RuntimeBundleError("runtime_bundle_archive_multi_disk_unsupported")
    if records_on_disk != total_records:
        raise RuntimeBundleError("runtime_bundle_archive_record_count_mismatch")
    if total_records > MAXIMUM_CENTRAL_DIRECTORY_RECORDS:
        raise RuntimeBundleError("runtime_bundle_archive_too_many_records")

    if directory_offset > eocd or directory_size > eocd:
        raise RuntimeBundleError("runtime_bundle_archive_directory_out_of_bounds")
    if directory_offset + directory_size != eocd:
        raise RuntimeBundleError("runtime_bundle_archive_directory_extent_invalid")

    names: list[str] = []
    cursor = directory_offset
    limit = directory_offset + directory_size
    for _ in range(total_records):
        if cursor + CENTRAL_DIRECTORY_FIXED_BYTES > limit:
            raise RuntimeBundleError("runtime_bundle_archive_directory_truncated")
        if payload[cursor : cursor + 4] != CENTRAL_DIRECTORY_SIGNATURE:
            raise RuntimeBundleError("runtime_bundle_archive_directory_signature_invalid")

        flags = _u16(payload, cursor + 8)
        name_length = _u16(payload, cursor + 28)
        extra_length = _u16(payload, cursor + 30)
        comment_bytes = _u16(payload, cursor + 32)
        if name_length == 0:
            raise RuntimeBundleError("runtime_bundle_member_name_invalid")

        start = cursor + CENTRAL_DIRECTORY_FIXED_BYTES
        end = start + name_length
        record_end = end + extra_length + comment_bytes
        if record_end > limit:
            raise RuntimeBundleError("runtime_bundle_archive_directory_truncated")

        raw = payload[start:end]
        try:
            names.append(raw.decode("utf-8" if flags & UTF8_NAME_FLAG else "cp437"))
        except UnicodeDecodeError:
            raise RuntimeBundleError("runtime_bundle_member_name_undecodable") from None
        cursor = record_end

    if cursor != limit:
        raise RuntimeBundleError("runtime_bundle_archive_directory_extent_invalid")
    return names


# --------------------------------------------------------------------------------- archive checks


def _refuse_unsafe_member_name(name: str) -> None:
    """Reject any member name that could escape the artifact tree or resolve ambiguously."""

    if not name or name != name.strip():
        raise RuntimeBundleError("runtime_bundle_member_name_invalid")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in name):
        raise RuntimeBundleError("runtime_bundle_member_name_invalid")
    if "\\" in name:
        raise RuntimeBundleError("runtime_bundle_member_backslash")
    if name.startswith("/") or name.startswith("./"):
        raise RuntimeBundleError("runtime_bundle_member_absolute")
    # A Windows drive or UNC prefix is absolute even though it contains no leading slash.
    if len(name) >= 2 and name[1] == ":":
        raise RuntimeBundleError("runtime_bundle_member_absolute")
    if name.endswith("/"):
        raise RuntimeBundleError("runtime_bundle_member_directory")

    components = name.split("/")
    if ".." in components:
        raise RuntimeBundleError("runtime_bundle_member_traversal")
    # An empty component is a repeated separator; a "." component is a redundant self-reference.
    # Neither can appear in an approved name, and both make two spellings of one path.
    if "" in components or "." in components:
        raise RuntimeBundleError("runtime_bundle_member_name_invalid")


def _refuse_unsafe_member_mode(info: zipfile.ZipInfo) -> None:
    """Reject non-regular members: symlinks, devices, FIFOs, sockets."""

    if info.is_dir():
        raise RuntimeBundleError("runtime_bundle_member_directory")
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise RuntimeBundleError("runtime_bundle_member_symlink")
    # A zero file type means the archive recorded no Unix mode at all, which is ordinary for
    # archives written on Windows and is not by itself a non-regular member.
    if file_type not in (0, stat.S_IFREG):
        raise RuntimeBundleError("runtime_bundle_member_not_regular")


def _refuse_unsupported_member(info: zipfile.ZipInfo) -> None:
    """Reject encrypted members and any compression method this bundle does not produce."""

    if info.flag_bits & 0x1:
        raise RuntimeBundleError("runtime_bundle_member_encrypted")
    # Bit 3 means sizes live in a trailing data descriptor rather than the local header, which makes
    # the declared size ambiguous. This bundle never writes one.
    if info.flag_bits & 0x8:
        raise RuntimeBundleError("runtime_bundle_member_data_descriptor_unsupported")
    if info.compress_type != ZIP_COMPRESSION:
        raise RuntimeBundleError("runtime_bundle_member_compression_unsupported")


def _validate_bundle_manifest(manifest: dict[str, Any], artifacts: dict[str, bytes]) -> None:
    """Check the archive's own claims before trusting any of its bytes."""

    if set(manifest) != {
        "artifact_type",
        "artifacts",
        "bundle_schema_version",
        "members",
        "product",
        "release_tag",
        "source",
    }:
        raise RuntimeBundleError("runtime_bundle_manifest_schema_invalid")
    if manifest.get("product") != PRODUCT:
        raise RuntimeBundleError("runtime_bundle_manifest_product_invalid")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeBundleError("runtime_bundle_manifest_artifact_type_invalid")
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise RuntimeBundleError("runtime_bundle_manifest_schema_invalid")
    if manifest.get("release_tag") != RELEASE_TAG:
        raise RuntimeBundleError("runtime_bundle_manifest_release_invalid")
    if manifest.get("members") != sorted(BUNDLE_MEMBERS):
        raise RuntimeBundleError("runtime_bundle_manifest_members_invalid")

    source = manifest.get("source")
    if not isinstance(source, dict) or set(source) != {"feature_dataset_id", "model_id"}:
        raise RuntimeBundleError("runtime_bundle_manifest_source_invalid")

    declared = manifest.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(BUNDLE_MEMBERS):
        raise RuntimeBundleError("runtime_bundle_manifest_artifacts_invalid")

    for member in BUNDLE_MEMBERS:
        entry = declared.get(member)
        if not isinstance(entry, dict) or set(entry) != {"byte_size", "sha256"}:
            raise RuntimeBundleError("runtime_bundle_manifest_artifacts_invalid")
        payload = artifacts[member]
        if entry["byte_size"] != len(payload):
            raise RuntimeBundleError("runtime_bundle_declared_size_mismatch")
        actual = _sha256(payload)
        if entry["sha256"] != actual:
            raise RuntimeBundleError("runtime_bundle_declared_hash_mismatch")
        # The pinned value is the authority. A bundle whose manifest agrees with its own tampered
        # payload still fails here.
        if actual != EXPECTED_SHA256[member]:
            raise RuntimeBundleError("runtime_bundle_artifact_hash_mismatch")

    # The source identities must be the ones the approved manifests actually carry, so a bundle
    # cannot claim to descend from a different model than the bytes it ships.
    model_manifest = _canonical_json(artifacts[f"{MODEL_DIRECTORY}/manifest.json"])
    feature_manifest = _canonical_json(artifacts[f"{FEATURE_DIRECTORY}/manifest.json"])
    if source["model_id"] != model_manifest.get("model_id"):
        raise RuntimeBundleError("runtime_bundle_manifest_source_invalid")
    if source["feature_dataset_id"] != feature_manifest.get("feature_dataset_id"):
        raise RuntimeBundleError("runtime_bundle_manifest_source_invalid")


def _read_bounded_archive(archive_path: Path) -> bytes:
    """Read the archive with a hard ceiling on what can ever enter memory.

    ``stat()`` followed by ``Path.read_bytes()`` was check-then-act on size: the size was bounded,
    and the read that followed was not. A file that passed the check and then grew would be read in
    full, so the bound described an earlier moment rather than the allocation.

    The read is therefore capped at ``MAXIMUM_ARCHIVE_BYTES + 1`` on the same descriptor that was
    opened, and the one extra byte is what distinguishes "exactly at the ceiling" from "over it".
    ``fstat`` on that descriptor rejects an oversized file early, before the read, but it is an
    optimisation rather than the guarantee -- the guarantee is the capped read.

    This does not eliminate local check/use races and is not claimed to: the file can still change
    between the symlink check and this open. What it removes is the unbounded allocation, so the
    worst a racing writer can achieve is one byte past the documented ceiling.
    """

    try:
        with archive_path.open("rb") as handle:
            reported = os.fstat(handle.fileno()).st_size
            if reported > MAXIMUM_ARCHIVE_BYTES:
                raise RuntimeBundleError("runtime_bundle_archive_too_large")
            payload = handle.read(MAXIMUM_ARCHIVE_BYTES + 1)
    except OSError:
        raise RuntimeBundleError("runtime_bundle_archive_unreadable") from None

    # Authoritative, because it measures what was actually delivered rather than what was declared.
    if len(payload) > MAXIMUM_ARCHIVE_BYTES:
        raise RuntimeBundleError("runtime_bundle_archive_too_large")
    if not payload:
        raise RuntimeBundleError("runtime_bundle_archive_empty")
    return payload


def read_validated_archive(archive_path: Path) -> dict[str, bytes]:
    """Validate an archive completely and return its approved artifact bytes.

    Nothing is written to the filesystem by this function. ``extractall`` is deliberately not used:
    every member is inspected, size-checked and read individually.
    """

    # No boundary: this path arrives from outside any tree we own, so the walk runs to the
    # filesystem anchor. No ``resolve()`` first -- that would follow the link we must reject.
    _refuse_symlinked(archive_path)
    if not archive_path.is_file():
        raise RuntimeBundleError("runtime_bundle_archive_missing")

    stored_payload = _read_bounded_archive(archive_path)

    # The outer pin, and the first thing any byte of this file meets. Everything below -- the EOCD
    # arithmetic, the central-directory walk, ``zipfile``, the manifest reader -- only ever runs on
    # bytes that already matched a source-code constant. Appended, prepended, truncated or otherwise
    # reshaped archives stop here rather than in format-parsing code.
    if _sha256(stored_payload) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeBundleError("runtime_bundle_archive_hash_mismatch")

    # Structural parse, against the stored bytes rather than the normalised view.
    stored_names = raw_member_names(stored_payload)
    if len(stored_names) != len(set(stored_names)):
        raise RuntimeBundleError("runtime_bundle_member_duplicate")
    for stored_name in stored_names:
        _refuse_unsafe_member_name(stored_name)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeBundleError("runtime_bundle_member_duplicate")
            # The raw parse and zipfile's own view must describe the same archive. A disagreement
            # means one of the two was misled about the record set.
            if sorted(names) != sorted(stored_names):
                raise RuntimeBundleError("runtime_bundle_archive_directory_inconsistent")

            for info in infos:
                _refuse_unsafe_member_name(info.filename)
                _refuse_unsafe_member_mode(info)
                _refuse_unsupported_member(info)
                if info.filename not in ARCHIVE_NAMES:
                    raise RuntimeBundleError("runtime_bundle_member_unknown")
                if info.file_size > MAXIMUM_MEMBER_BYTES:
                    raise RuntimeBundleError("runtime_bundle_member_too_large")

            if set(names) != set(ARCHIVE_NAMES):
                raise RuntimeBundleError("runtime_bundle_member_missing")
            if sum(info.file_size for info in infos) > MAXIMUM_TOTAL_BYTES:
                raise RuntimeBundleError("runtime_bundle_total_too_large")

            payloads = {name: _read_member(archive, name) for name in sorted(ARCHIVE_NAMES)}
    except zipfile.BadZipFile:
        # Covers a bad CRC, a truncated member, and a central-directory/local-header name
        # disagreement, all of which zipfile detects for us while reading.
        raise RuntimeBundleError("runtime_bundle_archive_corrupt") from None
    except OSError:
        raise RuntimeBundleError("runtime_bundle_archive_unreadable") from None

    manifest = _canonical_json(payloads.pop(BUNDLE_MANIFEST_NAME))
    for member in BUNDLE_MEMBERS:
        _canonical_json(payloads[member])
    _validate_bundle_manifest(manifest, payloads)
    return payloads


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read one member, with size bounded and the CRC checked by ``zipfile`` at EOF."""

    try:
        with archive.open(name) as stream:
            payload = stream.read(MAXIMUM_MEMBER_BYTES + 1)
    except RuntimeError:
        # zipfile raises a bare RuntimeError for a password-protected member.
        raise RuntimeBundleError("runtime_bundle_member_encrypted") from None
    if len(payload) > MAXIMUM_MEMBER_BYTES:
        raise RuntimeBundleError("runtime_bundle_member_too_large")
    if not payload:
        raise RuntimeBundleError("runtime_bundle_member_empty")
    return payload


# --------------------------------------------------------------------------------- download


def _refuse_unsafe_download_url(url: str) -> None:
    """Permit HTTPS on the release hosts only. There is no caller-supplied URL to widen this.

    A query string is permitted because GitHub's asset redirect carries a signed one; it is never
    logged. Everything else that can change where or how the request goes is refused.
    """

    parts = urlsplit(url)
    if parts.scheme != "https":
        raise RuntimeBundleError("runtime_bundle_download_scheme_invalid")
    if parts.username is not None or parts.password is not None:
        raise RuntimeBundleError("runtime_bundle_download_userinfo_rejected")
    if parts.fragment:
        raise RuntimeBundleError("runtime_bundle_download_fragment_rejected")
    try:
        port = parts.port
    except ValueError:
        raise RuntimeBundleError("runtime_bundle_download_port_invalid") from None
    if port not in PERMITTED_DOWNLOAD_PORTS:
        raise RuntimeBundleError("runtime_bundle_download_port_invalid")
    if parts.hostname is None or parts.hostname.lower() not in PERMITTED_DOWNLOAD_HOSTS:
        raise RuntimeBundleError("runtime_bundle_download_host_invalid")


def download_release_asset(destination: Path, *, client: httpx.Client | None = None) -> Path:
    """Download the one fixed release asset.

    Takes no URL, and no environment variable or configuration field can redirect it. Redirects are
    followed manually so that every hop is checked against the permitted host set rather than only
    the first one. No response body, header, query string or signed URL is ever logged. A partial
    download is removed rather than left behind looking like an archive.
    """

    owned = client is None
    session = client or httpx.Client(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        url = RELEASE_URL
        for _ in range(MAXIMUM_REDIRECTS + 1):
            _refuse_unsafe_download_url(url)
            try:
                with session.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeBundleError("runtime_bundle_download_redirect_invalid")
                        url = str(response.url.join(location))
                        continue
                    if response.status_code != 200:
                        raise RuntimeBundleError("runtime_bundle_download_failed")
                    _refuse_oversized_declared_length(response)
                    _stream_to_file(response, destination)
                    return destination
            except httpx.HTTPError:
                raise RuntimeBundleError("runtime_bundle_download_failed") from None
        raise RuntimeBundleError("runtime_bundle_download_too_many_redirects")
    finally:
        if owned:
            session.close()


def _refuse_oversized_declared_length(response: httpx.Response) -> None:
    """Refuse before streaming when the server already declares an oversized body."""

    declared = response.headers.get("content-length")
    if declared is None:
        return
    try:
        length = int(declared)
    except ValueError:
        raise RuntimeBundleError("runtime_bundle_download_length_invalid") from None
    if length < 0:
        raise RuntimeBundleError("runtime_bundle_download_length_invalid")
    if length > MAXIMUM_ARCHIVE_BYTES:
        raise RuntimeBundleError("runtime_bundle_download_too_large")


def _stream_to_file(response: httpx.Response, destination: Path) -> None:
    """Write a bounded response body into a destination this attempt exclusively created.

    The cap is enforced on bytes actually received, not on the declared length, so a server that
    lies about ``Content-Length`` or omits it entirely is still bounded.

    An existing destination is never truncated, overwritten or removed: the exclusive create fails
    outright and no byte of the response is written. Cleanup is therefore unambiguous -- a file is
    removed on failure exactly when this attempt is the one that brought it into existence.
    """

    # Ownership is acquired atomically, not inferred. The previous version asked ``exists()`` and
    # then opened ``"wb"``, which is two defects in one: ``wb`` truncates, so an existing file was
    # already destroyed by the time the guard declined to delete it, and the answer to ``exists()``
    # could be stale by the time the open happened. ``O_CREAT | O_EXCL`` collapses both into one
    # atomic acquisition -- the file is ours because the kernel says we created it, or it is not
    # ours and we never touch a byte of it.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        raise RuntimeBundleError("runtime_bundle_download_destination_exists") from None
    except OSError:
        raise RuntimeBundleError("runtime_bundle_download_failed") from None

    # Only now, and only because the exclusive create succeeded.
    owned = True
    written = 0
    completed = False
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except OSError:
            # Wrapping failed, so the descriptor is still raw and still ours to close.
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise RuntimeBundleError("runtime_bundle_download_failed") from None

        try:
            with handle:
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > MAXIMUM_ARCHIVE_BYTES:
                        raise RuntimeBundleError("runtime_bundle_download_too_large")
                    handle.write(chunk)
        except OSError:
            raise RuntimeBundleError("runtime_bundle_download_failed") from None
        if written == 0:
            raise RuntimeBundleError("runtime_bundle_download_empty")
        completed = True
    finally:
        # ``finally`` rather than a list of exception types. An earlier version caught only
        # ``RuntimeBundleError``, and ``httpx.HTTPError`` is not an ``OSError``, so a connection
        # dropped mid-stream left a truncated file sitting there looking like an archive. This
        # covers transport errors, size limits, cancellation and anything else that exits early.
        if owned and not completed:
            # The original failure is the useful one; a cleanup error must never replace it.
            with contextlib.suppress(OSError):
                destination.unlink(missing_ok=True)


# --------------------------------------------------------------------------------- install


def _staging_directory(root: Path) -> Path:
    """A process-owned staging directory on the destination filesystem.

    Created beside the artifact tree so publication is a same-filesystem rename rather than a copy
    that can fail halfway. ``mkdtemp`` creates it 0o700 and owned by this process.
    """

    try:
        return Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=root))
    except OSError:
        raise RuntimeBundleError("runtime_bundle_staging_failed") from None


def _remove_staging(staging: Path, root: Path) -> None:
    """Remove only our own staging directory, and prove it is ours before removing it."""

    if not staging.exists():
        return
    if staging.parent != root or not staging.name.startswith(STAGING_PREFIX):
        raise RuntimeBundleError("runtime_bundle_staging_cleanup_unsafe")
    shutil.rmtree(staging, ignore_errors=True)


def _refuse_conflicting_model_directory(root: Path, members: tuple[str, ...]) -> None:
    """Refuse a model directory whose final contents would fail ``load_locked_model``.

    That loader requires the directory to hold exactly its three artifacts. An unrelated file left
    beside them makes startup fail with ``locked_model_artifact_set_invalid``, so installing over
    it would produce a bundle that verifies here and refuses to serve.
    """

    directory = root / MODEL_DIRECTORY
    if not directory.is_dir():
        return
    try:
        existing = {entry.name for entry in directory.iterdir()}
    except OSError:
        raise RuntimeBundleError("runtime_bundle_destination_unreadable") from None
    incoming = {
        member.rsplit("/", 1)[1] for member in members if member.startswith(MODEL_DIRECTORY)
    }
    if (existing | incoming) - set(MODEL_ARTIFACT_FILENAMES):
        raise RuntimeBundleError("runtime_bundle_model_directory_has_unknown_files")


def _classify_destination(path: Path, payload: bytes, root: Path) -> bool:
    """Return True when the file already holds exactly these bytes.

    A destination observed to hold *different* content is never overwritten: these are locked
    artifacts, and silently replacing one would destroy the very property the hashes exist to
    guarantee. Qualified deliberately -- this is a classification, and publication happens after it,
    so the guarantee is "never deliberately overwrites content it saw differ", not a lock. See the
    module docstring for the local-host trust boundary that makes the difference immaterial.
    """

    _refuse_symlinked(path, boundary=root)
    if not path.exists():
        return False
    if not path.is_file():
        raise RuntimeBundleError("runtime_bundle_destination_not_a_file")
    try:
        existing = path.read_bytes()
    except OSError:
        raise RuntimeBundleError("runtime_bundle_destination_unreadable") from None
    if existing != payload:
        raise RuntimeBundleError("runtime_bundle_destination_conflict")
    return True


def _publication_order(members: tuple[str, ...]) -> Iterator[str]:
    """Data files first, then the manifest that describes them, per directory."""

    yield from sorted(
        members, key=lambda name: (name.rsplit("/", 1)[0], name.endswith("manifest.json"), name)
    )


def _roll_back(created: list[Path], created_directories: list[Path]) -> None:
    """Undo exactly what this attempt created, and nothing else.

    Scoped deliberately narrowly: only paths this attempt wrote are removed, and only directories
    this attempt created and that are still empty. A pre-existing identical file was never touched
    and is never removed; a conflicting file was refused before any write, so it cannot appear here.

    A multi-directory publication is not an atomic transaction and this is not claimed to be one.
    It is a best-effort return to the pre-install state for the failure shapes that can actually
    occur mid-publication, and it is proven per step by failure-injection tests.
    """

    for path in reversed(created):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Leaving a file behind is worse than silent, so the caller's error stands and the
            # residual state is documented rather than masked by a second failure.
            continue
    for directory in reversed(created_directories):
        try:
            directory.rmdir()
        except OSError:
            continue


def install_bundle(
    artifacts: dict[str, bytes],
    *,
    root: Path | None = None,
    require_evaluation: bool = True,
) -> InstallationResult:
    """Publish validated artifacts, then prove the ordinary serving binding still succeeds."""

    destination_root = REPOSITORY_ROOT if root is None else _absolute(root)
    members = BUNDLE_MEMBERS if require_evaluation else STARTUP_MEMBERS
    if any(member not in artifacts for member in members):
        raise RuntimeBundleError("runtime_bundle_member_missing")

    _refuse_conflicting_model_directory(destination_root, members)

    installed: list[str] = []
    already: list[str] = []
    created: list[Path] = []
    created_directories: list[Path] = []
    staging = _staging_directory(destination_root)
    try:
        # Every destination is classified before anything is written, so a conflict anywhere
        # refuses the whole installation rather than being discovered halfway through.
        pending: list[str] = []
        for member in _publication_order(members):
            destination = destination_root / member
            if _classify_destination(destination, artifacts[member], destination_root):
                already.append(member)
            else:
                pending.append(member)

        for member in pending:
            destination = destination_root / member
            staged = staging / member.replace("/", "__")
            try:
                staged.write_bytes(artifacts[member])
                for parent in reversed(destination.parent.parents):
                    if parent.is_relative_to(destination_root) and not parent.exists():
                        parent.mkdir()
                        created_directories.append(parent)
                if not destination.parent.exists():
                    destination.parent.mkdir(parents=True)
                    created_directories.append(destination.parent)
                _refuse_symlinked(destination.parent, boundary=destination_root)
                os.replace(staged, destination)
            except OSError:
                _roll_back(created, created_directories)
                raise RuntimeBundleError("runtime_bundle_publication_failed") from None
            created.append(destination)
            installed.append(member)
    except RuntimeBundleError:
        _roll_back(created, created_directories)
        raise
    finally:
        _remove_staging(staging, destination_root)

    try:
        verify_installation(root=destination_root, require_evaluation=require_evaluation)
    except RuntimeBundleError:
        # The bytes are right or they would have failed validation, so a binding failure means the
        # environment cannot serve them. Leave what was installed in place -- rolling back here
        # would delete correct artifacts and make the real problem harder to see -- and fail closed.
        raise

    return InstallationResult(
        installed=tuple(installed),
        already_present=tuple(already),
        evaluation_installed=EVALUATION_MEMBER in members,
    )


# --------------------------------------------------------------------------------- verify


def verify_installation(*, root: Path | None = None, require_evaluation: bool = False) -> None:
    """Check installed hashes, then run the application's own startup binding.

    The binding check is the point. Matching hashes prove the right bytes are present; only
    ``load_serving_bundle`` proves the service will actually start with them.

    The evaluation artifact is verified by hash when the complete bundle is requested, but it is
    deliberately not part of the binding: the service starts without it.
    """

    destination_root = REPOSITORY_ROOT if root is None else _absolute(root)
    members = BUNDLE_MEMBERS if require_evaluation else STARTUP_MEMBERS

    for member in members:
        path = destination_root / member
        _refuse_symlinked(path, boundary=destination_root)
        if not path.is_file():
            raise RuntimeBundleError("runtime_bundle_installed_missing")
        try:
            payload = path.read_bytes()
        except OSError:
            raise RuntimeBundleError("runtime_bundle_installed_unreadable") from None
        if _sha256(payload) != EXPECTED_SHA256[member]:
            raise RuntimeBundleError("runtime_bundle_installed_hash_mismatch")

    try:
        load_serving_bundle(
            feature_config_path=destination_root / "configs/features/default.json",
            modeling_config_path=destination_root / "configs/modeling/default.json",
            model_directory=destination_root / MODEL_DIRECTORY,
            feature_manifest_path=destination_root / f"{FEATURE_DIRECTORY}/manifest.json",
        )
    except ServingBundleError as error:
        raise RuntimeBundleError(f"runtime_bundle_binding_failed:{error}") from None


def install_from_archive(
    archive_path: Path,
    *,
    root: Path | None = None,
    require_evaluation: bool = True,
) -> InstallationResult:
    """Validate an archive on disk and install it. Performs no network access."""

    return install_bundle(
        read_validated_archive(archive_path),
        root=root,
        require_evaluation=require_evaluation,
    )


def install_from_release(
    *,
    root: Path | None = None,
    require_evaluation: bool = True,
    client: httpx.Client | None = None,
) -> InstallationResult:
    """Download the fixed release asset into a process-owned temporary file and install it."""

    try:
        directory = Path(tempfile.mkdtemp(prefix=DOWNLOAD_PREFIX))
    except OSError:
        raise RuntimeBundleError("runtime_bundle_staging_failed") from None
    archive_path = directory / ASSET_NAME
    try:
        download_release_asset(archive_path, client=client)
        return install_from_archive(
            archive_path,
            root=root,
            require_evaluation=require_evaluation,
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)
