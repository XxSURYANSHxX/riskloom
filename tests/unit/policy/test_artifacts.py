import json
from pathlib import Path

import pytest

from riskloom.policy.artifacts import (
    policy_config_sha256,
    publish_band,
    publish_comparison,
    require_output_separate,
)
from riskloom.policy.canonical import (
    PolicyArtifactError,
    canonical_json_bytes,
    canonical_sha256,
    read_canonical_json,
)
from riskloom.policy.config import PolicyConfig, load_policy_config


@pytest.fixture(scope="session")
def policy_config() -> PolicyConfig:
    return load_policy_config(Path("configs/policy/default.json"))


BAND_PAYLOAD = {
    "artifact_type": "cost_aware_policy_band",
    "band": {"lower_threshold": 0.1, "upper_threshold": 0.4},
    "model_id": "a" * 64,
    "product": "RiskLoom",
}
COMPARISON_PAYLOAD = {"gates": {"approval_eligible": False}, "product": "RiskLoom"}
SOURCE = {"model_id": "a" * 64}


def test_band_publication_is_canonical_manifest_last_and_hashes_only_the_band(
    tmp_path: Path, policy_config: PolicyConfig
) -> None:
    output = tmp_path / "band"
    result = publish_band(output, BAND_PAYLOAD, policy_config, SOURCE)
    assert {path.name for path in output.iterdir()} == {"band.json", "manifest.json"}
    manifest = read_canonical_json(output / "manifest.json")
    band = read_canonical_json(output / "band.json")
    # The manifest hashes the band and never itself.
    assert set(manifest["artifacts"]) == {"band.json"}
    assert manifest["artifacts"]["band.json"]["sha256"] == canonical_sha256(band)
    assert "manifest.json" not in manifest["artifacts"]
    assert not any("manifest" in key and "sha256" in key for key in manifest)
    assert manifest["band_id"] == result.artifact_id == band["band_id"]
    assert manifest["effective_policy_configuration_sha256"] == policy_config_sha256(policy_config)
    raw = (output / "band.json").read_bytes()
    assert raw == canonical_json_bytes(band)
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert json.loads(raw)


def test_comparison_publication_is_canonical_and_binds_its_source(
    tmp_path: Path, policy_config: PolicyConfig
) -> None:
    output = tmp_path / "comparison"
    result = publish_comparison(output, COMPARISON_PAYLOAD, policy_config, SOURCE)
    assert {path.name for path in output.iterdir()} == {"comparison.json", "manifest.json"}
    manifest = read_canonical_json(output / "manifest.json")
    comparison = read_canonical_json(output / "comparison.json")
    assert set(manifest["artifacts"]) == {"comparison.json"}
    assert manifest["artifacts"]["comparison.json"]["sha256"] == canonical_sha256(comparison)
    assert manifest["comparison_id"] == result.artifact_id == comparison["comparison_id"]
    assert manifest["source"] == SOURCE


def test_publication_refuses_a_non_empty_destination(
    tmp_path: Path, policy_config: PolicyConfig
) -> None:
    output = tmp_path / "band"
    publish_band(output, BAND_PAYLOAD, policy_config, SOURCE)
    with pytest.raises(PolicyArtifactError, match="not_empty"):
        publish_band(output, BAND_PAYLOAD, policy_config, SOURCE)
    with pytest.raises(PolicyArtifactError, match="not_empty"):
        publish_comparison(output, COMPARISON_PAYLOAD, policy_config, SOURCE)


def test_publication_refuses_files_and_unsafe_roots(
    tmp_path: Path, policy_config: PolicyConfig
) -> None:
    target = tmp_path / "target"
    target.write_text("unknown", encoding="utf-8")
    with pytest.raises(PolicyArtifactError, match="not_directory"):
        publish_band(target, BAND_PAYLOAD, policy_config, SOURCE)
    with pytest.raises(PolicyArtifactError, match="unsafe"):
        publish_band(Path.cwd(), BAND_PAYLOAD, policy_config, SOURCE)
    with pytest.raises(PolicyArtifactError, match="unsafe"):
        publish_band(Path.home(), BAND_PAYLOAD, policy_config, SOURCE)


def test_publication_refuses_source_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(PolicyArtifactError, match="path_overlap"):
        require_output_separate(source / "nested", (source,))
    with pytest.raises(PolicyArtifactError, match="path_overlap"):
        require_output_separate(tmp_path, (source,))


def test_publication_leaves_no_staging_directory(
    tmp_path: Path, policy_config: PolicyConfig
) -> None:
    publish_band(tmp_path / "band", BAND_PAYLOAD, policy_config, SOURCE)
    assert {path.name for path in tmp_path.iterdir()} == {"band"}


def test_canonical_helpers_reject_uncanonical_and_unserialisable_input(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(b'{\n  "a": 1\n}\n')
    with pytest.raises(PolicyArtifactError, match="not_canonical"):
        read_canonical_json(path)
    path.write_bytes(b"not json")
    with pytest.raises(PolicyArtifactError, match="json_invalid"):
        read_canonical_json(path)
    with pytest.raises(PolicyArtifactError, match="not_canonicalizable"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(PolicyArtifactError, match="not_canonicalizable"):
        canonical_json_bytes({"value": object()})
