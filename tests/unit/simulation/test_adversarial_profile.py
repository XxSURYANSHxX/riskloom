"""The adversarial-stress profile, and the guarantee that it cannot touch anything else.

The most important tests in this file are the negative ones. Adding a field to the generator
configuration is the single most dangerous change available in this repository: every PRNG stream in
schema 1.1.0 is namespaced by a fingerprint computed over the whole configuration, so a stray key
would silently change the development dataset the locked Day 4 model was trained on.
"""

import json
from pathlib import Path

import pytest

from riskloom.simulation.config import (
    GeneratorConfig,
    configuration_fingerprint,
    effective_configuration,
    generator_version_for_config,
)

CONFIGS = Path("configs/simulation")
VARIANTS = ("slow-low", "window-edge", "distributed", "failure-camouflage")

# Recorded from the artifacts these configurations produced, before this gate existed.
LOCKED_FINGERPRINTS = {
    "development": "140ecd643528fadc5583957a07a4ee50452b0cac5c3a92a20aa41431320ca42c",
    "policy-validation": "88ec3bd4ee9540b2afb6d313ae23b1d731479444b2c63bd84ecacb97b90bfc26",
    "smoke": None,
}


def load(name: str) -> GeneratorConfig:
    return GeneratorConfig.model_validate(json.loads((CONFIGS / f"{name}.json").read_text()))


def raw(name: str) -> dict:
    return json.loads((CONFIGS / f"{name}.json").read_text())


# --------------------------------------------------------------------------- immutability


@pytest.mark.parametrize(("name", "expected"), sorted(LOCKED_FINGERPRINTS.items(), key=str))
def test_existing_configuration_fingerprints_are_unchanged(name: str, expected: str | None) -> None:
    """The single most important assertion in this gate.

    If either fingerprint moves, every identifier and every drawn value in that dataset moves with
    it, and the locked model's training data no longer exists.
    """

    assert configuration_fingerprint(load(name)) == expected


@pytest.mark.parametrize("name", ["development", "policy-validation", "smoke"])
def test_the_evasion_field_never_enters_an_older_canonical_configuration(name: str) -> None:
    for split in effective_configuration(load(name))["splits"]:
        assert "evasion_shape" not in split


def test_schema_1_2_0_maps_to_the_unchanged_1_1_0_algorithm() -> None:
    """The derivation did not change, so the algorithm version must not claim it did."""

    assert generator_version_for_config(load("adversarial-stress-slow-low")) == "1.1.0"
    assert generator_version_for_config(load("development")) == "1.1.0"
    assert generator_version_for_config(load("smoke")) == "1.0.0"


# --------------------------------------------------------------------------- contract isolation


@pytest.mark.parametrize("profile", ["development", "smoke", "policy-validation"])
def test_an_older_profile_may_not_carry_an_evasion_shape(profile: str) -> None:
    payload = raw(profile)
    payload["splits"][2]["evasion_shape"] = {"variant": "slow_and_low", "duration_minutes": 1440}
    with pytest.raises(ValueError, match="evasion_shape_requires"):
        GeneratorConfig.model_validate(payload)


def test_the_adversarial_profile_requires_schema_1_2_0() -> None:
    payload = raw("adversarial-stress-slow-low")
    payload["config_schema_version"] = "1.1.0"
    with pytest.raises(ValueError, match="evasion_shape_requires_configuration_schema_1_2_0"):
        GeneratorConfig.model_validate(payload)


def test_schema_1_2_0_is_reserved_for_the_adversarial_profile() -> None:
    payload = raw("development")
    payload["config_schema_version"] = "1.2.0"
    with pytest.raises(ValueError, match="reserved_for_adversarial_stress"):
        GeneratorConfig.model_validate(payload)


def test_relabelling_development_as_adversarial_fails_the_contract() -> None:
    """Renaming a profile must not be a way to smuggle a dataset past its contract."""

    payload = raw("development")
    payload["dataset_profile"] = "adversarial-stress"
    payload["config_schema_version"] = "1.2.0"
    with pytest.raises(ValueError, match="adversarial_contract_"):
        GeneratorConfig.model_validate(payload)


def test_relabelling_adversarial_as_development_fails_the_contract() -> None:
    payload = raw("adversarial-stress-slow-low")
    payload["dataset_profile"] = "development"
    with pytest.raises(ValueError):
        GeneratorConfig.model_validate(payload)


def test_the_baseline_splits_must_stay_baseline() -> None:
    """Evasion on train or calibration would break the dataset's own entity-shift invariant."""

    payload = raw("adversarial-stress-slow-low")
    payload["splits"][0]["evasion_shape"] = {"variant": "slow_and_low", "duration_minutes": 1440}
    with pytest.raises(ValueError, match="baseline_splits_must_stay_baseline"):
        GeneratorConfig.model_validate(payload)


def test_the_test_split_must_carry_an_evasion_shape() -> None:
    payload = raw("adversarial-stress-slow-low")
    payload["splits"][2].pop("evasion_shape")
    with pytest.raises(ValueError, match="test_evasion_required"):
        GeneratorConfig.model_validate(payload)


@pytest.mark.parametrize(
    "shape",
    [
        {"variant": "slow_and_low", "network_count": 4},
        {"variant": "distributed_thin", "duration_minutes": 1440},
        {"variant": "window_edge", "edge_window_seconds": 120},
        {"variant": "slow_and_low"},
        {"variant": "slow_and_low", "duration_minutes": 1440, "network_count": 4},
        {"variant": "invented_variant", "duration_minutes": 1440},
        {"variant": "slow_and_low", "duration_minutes": 90},
        {"variant": "slow_and_low", "duration_minutes": True},
        {"variant": "slow_and_low", "duration_minutes": 1440.5},
        {"variant": "slow_and_low", "duration_minutes": -1440},
        {"variant": "slow_and_low", "duration_minutes": 1440, "unknown_field": 1},
    ],
)
def test_a_malformed_evasion_shape_is_refused(shape: dict) -> None:
    payload = raw("adversarial-stress-slow-low")
    payload["splits"][2]["evasion_shape"] = shape
    with pytest.raises(ValueError):
        GeneratorConfig.model_validate(payload)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_shipped_variant_config_validates(variant: str) -> None:
    config = load(f"adversarial-stress-{variant}")
    assert config.dataset_profile == "adversarial-stress"
    assert config.config_schema_version == "1.2.0"
    assert config.total_events == 9_000
    assert config.scenario_weights.attack == 200, "prevalence must match the reference rows"


def test_the_four_variants_are_four_distinct_datasets() -> None:
    fingerprints = {
        variant: configuration_fingerprint(load(f"adversarial-stress-{variant}"))
        for variant in VARIANTS
    }
    assert len(set(fingerprints.values())) == len(VARIANTS)
    assert set(fingerprints.values()).isdisjoint(
        {value for value in LOCKED_FINGERPRINTS.values() if value is not None}
    )


# --------------------------------------------------------------------------- generation branches


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate each variant once. About four seconds apiece.

    Generated here rather than read from `artifacts/` so the evasion branches are covered by the
    suite itself, not only on a machine where the manual datasets happen to exist.
    """

    from riskloom.simulation.artifacts import generate_dataset

    root = tmp_path_factory.mktemp("adversarial")
    outputs: dict[str, Path] = {}
    for variant in VARIANTS:
        target = root / variant
        generate_dataset(load(f"adversarial-stress-{variant}"), 20260921, target)
        outputs[variant] = target
    return outputs


def _test_attacks(directory: Path) -> list[dict]:
    events = {
        json.loads(line)["event_id"]: json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    }
    rows = []
    for line in (directory / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        label = json.loads(line)
        if label["scenario_type"] == "card_testing_campaign" and label["split"] == "test":
            row = dict(events[label["event_id"]])
            row["_campaign"] = label["campaign_id"]
            rows.append(row)
    return rows


def _gaps(rows: list[dict]) -> list[float]:
    from collections import defaultdict
    from datetime import datetime

    grouped: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        grouped[row["_campaign"]].append(
            datetime.fromisoformat(row["occurred_at"].replace("Z", "+00:00"))
        )
    gaps: list[float] = []
    for times in grouped.values():
        ordered = sorted(times)
        gaps.extend((b - a).total_seconds() for a, b in zip(ordered, ordered[1:], strict=False))
    return gaps


def test_window_edge_spaces_events_at_exactly_one_window(generated: dict[str, Path]) -> None:
    """The evasion this variant exists for: every prior event lands on the expiry cutoff."""

    import statistics

    assert statistics.median(_gaps(_test_attacks(generated["window-edge"]))) == pytest.approx(
        3_600, abs=1
    )


def test_slow_and_low_stretches_the_campaign(generated: dict[str, Path]) -> None:
    import statistics

    slow = statistics.median(_gaps(_test_attacks(generated["slow-low"])))
    baseline = statistics.median(_gaps(_test_attacks(generated["failure-camouflage"])))
    assert slow > baseline * 10


def test_distributed_thin_dilutes_devices_and_networks(generated: dict[str, Path]) -> None:
    thin = _test_attacks(generated["distributed"])
    baseline = _test_attacks(generated["failure-camouflage"])

    def per_network(rows: list[dict]) -> float:
        return len(rows) / max(len({r["network_token"] for r in rows if r["network_token"]}), 1)

    assert per_network(thin) < per_network(baseline)


def test_failure_camouflage_lowers_the_failure_rate(generated: dict[str, Path]) -> None:
    def rate(rows: list[dict]) -> float:
        return sum(1 for r in rows if r["failure_category"] is not None) / len(rows)

    assert rate(_test_attacks(generated["failure-camouflage"])) < 0.25
    assert rate(_test_attacks(generated["slow-low"])) > 0.5


def test_every_variant_keeps_the_same_attack_volume(generated: dict[str, Path]) -> None:
    counts = {v: len(_test_attacks(path)) for v, path in generated.items()}
    assert len(set(counts.values())) == 1, counts


def test_generation_is_deterministic(generated: dict[str, Path], tmp_path: Path) -> None:
    from riskloom.simulation.artifacts import generate_dataset

    repeat = tmp_path / "again"
    generate_dataset(load("adversarial-stress-window-edge"), 20260921, repeat)
    first = json.loads((generated["window-edge"] / "manifest.json").read_text())
    second = json.loads((repeat / "manifest.json").read_text())
    assert first["dataset_id"] == second["dataset_id"]


def test_variant_datasets_are_distinct_from_each_other(generated: dict[str, Path]) -> None:
    ids = {
        v: json.loads((path / "manifest.json").read_text())["dataset_id"]
        for v, path in generated.items()
    }
    assert len(set(ids.values())) == len(VARIANTS), ids
