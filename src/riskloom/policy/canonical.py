import hashlib
import json
from pathlib import Path
from typing import Any


class PolicyArtifactError(ValueError):
    """A safe policy artifact error carrying no source values."""


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
        raise PolicyArtifactError("policy_value_not_canonicalizable") from None
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
        raise PolicyArtifactError("policy_source_unreadable") from None
    return {"byte_size": byte_size, "sha256": digest.hexdigest()}


def read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PolicyArtifactError("policy_json_invalid") from None
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise PolicyArtifactError("policy_json_not_canonical")
    return value
