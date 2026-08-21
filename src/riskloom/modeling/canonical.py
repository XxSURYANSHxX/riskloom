import hashlib
import json
from pathlib import Path
from typing import Any


class ModelingArtifactError(ValueError):
    """A safe modeling artifact error with no source values."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise ModelingArtifactError("modeling_value_not_canonicalizable") from None
    return (rendered + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_metadata(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                byte_size += len(chunk)
                digest.update(chunk)
    except OSError:
        raise ModelingArtifactError("modeling_source_unreadable") from None
    return {"byte_size": byte_size, "sha256": digest.hexdigest()}


def read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ModelingArtifactError("modeling_json_invalid") from None
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ModelingArtifactError("modeling_json_not_canonical")
    return value


def parse_canonical_jsonl_row(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise ModelingArtifactError("modeling_jsonl_invalid") from None
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ModelingArtifactError("modeling_jsonl_not_canonical")
    return value
