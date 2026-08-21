from pathlib import Path

import pytest

from riskloom.modeling.canonical import (
    ModelingArtifactError,
    canonical_json_bytes,
    file_metadata,
    parse_canonical_jsonl_row,
    read_canonical_json,
)


def test_canonical_json_rejects_nonfinite_and_unsupported_values() -> None:
    with pytest.raises(ModelingArtifactError, match="not_canonicalizable"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(ModelingArtifactError, match="not_canonicalizable"):
        canonical_json_bytes({"value": object()})


def test_file_and_json_read_errors_are_safe(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ModelingArtifactError, match="source_unreadable"):
        file_metadata(missing)
    with pytest.raises(ModelingArtifactError, match="json_invalid"):
        read_canonical_json(missing)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"{\n")
    with pytest.raises(ModelingArtifactError, match="json_invalid"):
        read_canonical_json(invalid)


@pytest.mark.parametrize("raw", [b"[]\n", b'{"b":1, "a":2}\n', b'{"a":2,"b":1}'])
def test_json_and_jsonl_require_object_canonical_bytes(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(raw)
    with pytest.raises(ModelingArtifactError, match="not_canonical"):
        read_canonical_json(path)
    with pytest.raises(ModelingArtifactError, match="not_canonical"):
        parse_canonical_jsonl_row(raw)


def test_jsonl_invalid_syntax_is_safe() -> None:
    with pytest.raises(ModelingArtifactError, match="jsonl_invalid"):
        parse_canonical_jsonl_row(b"{\n")
