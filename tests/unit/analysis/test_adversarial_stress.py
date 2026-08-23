"""Scoring, reporting, and the guarantee that each variant actually reshaped its target.

The efficacy tests matter as much as the scoring ones. A variant that silently generated ordinary
traffic would report a comfortable null result that reads as model robustness, which is the most
expensive way this gate could be wrong.
"""

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from riskloom.analysis.adversarial_stress import (
    FALSE_NEGATIVE_COST_UNITS,
    FALSE_POSITIVE_COST_UNITS,
    VARIANTS,
    average_precision,
    build_report,
    write_report,
)
from riskloom.analysis.references import ReferenceRow

SIMULATIONS = Path("artifacts/simulations")
REPORT = Path("artifacts/analysis/adversarial-stress/adversarial_stress.json")

generated_only = pytest.mark.skipif(
    not all((SIMULATIONS / f"adversarial-{v}").exists() for v in VARIANTS),
    reason="adversarial datasets are generated manually and are Git-ignored",
)
report_only = pytest.mark.skipif(not REPORT.exists(), reason="report not generated")


def attacks_for(variant: str) -> list[dict]:
    directory = SIMULATIONS / f"adversarial-{variant}"
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


def campaign_gaps(rows: list[dict]) -> list[float]:
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


# --------------------------------------------------------------------------- average precision


def test_average_precision_is_one_for_a_perfect_ranking() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert average_precision(labels, scores) == pytest.approx(1.0)


def test_average_precision_matches_a_hand_computed_case() -> None:
    """Ranks: hit, miss, hit. Precision at the hits is 1/1 and 2/3, averaged over 2 positives."""

    labels = np.array([1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7])
    assert average_precision(labels, scores) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_is_none_without_positives() -> None:
    assert average_precision(np.array([0, 0]), np.array([0.5, 0.4])) is None


# --------------------------------------------------------------------------- variant efficacy


@generated_only
def test_slow_and_low_actually_slows_the_campaign_down() -> None:
    """Against failure-camouflage, which leaves timing untouched and so is the shape baseline."""

    slow = statistics.median(campaign_gaps(attacks_for("slow-low")))
    baseline = statistics.median(campaign_gaps(attacks_for("failure-camouflage")))
    assert slow > baseline * 10, (slow, baseline)


@generated_only
def test_window_edge_spaces_events_at_exactly_one_window() -> None:
    gaps = campaign_gaps(attacks_for("window-edge"))
    assert statistics.median(gaps) == pytest.approx(3_600, abs=1)


@generated_only
def test_distributed_thin_actually_dilutes_entity_reuse() -> None:
    rows = attacks_for("distributed")
    baseline = attacks_for("failure-camouflage")

    def per_network(sample: list[dict]) -> float:
        networks = {row["network_token"] for row in sample if row["network_token"]}
        return len(sample) / max(len(networks), 1)

    assert per_network(rows) < per_network(baseline)


@generated_only
def test_failure_camouflage_actually_lowers_the_failure_rate() -> None:
    def rate(sample: list[dict]) -> float:
        return sum(1 for row in sample if row["failure_category"] is not None) / len(sample)

    assert rate(attacks_for("failure-camouflage")) < rate(attacks_for("slow-low")) / 5


@generated_only
def test_every_variant_keeps_the_same_attack_volume() -> None:
    """Volume is held constant so a detection drop cannot be explained by fewer attacks."""

    counts = {variant: len(attacks_for(variant)) for variant in VARIANTS}
    assert len(set(counts.values())) == 1, counts


# --------------------------------------------------------------------------- dataset identity


@generated_only
def test_every_variant_dataset_is_distinct_from_every_locked_dataset() -> None:
    locked = {
        "development": "43586ca24c68f4b58cacde9444b3e9a54226a588db9e3de8244b1c7e92ad0ee1",
        "smoke": "5f8e96be454b50ea7c0389e9be6403dccd9209b4c21ba6647ed08d7871ee8181",
        "policy-validation": "e0adb284e232c08e7d5aa53a3c61ca420c0dd6b43689495f9d9371e4dd8ffdc6",
    }
    ids = {
        variant: json.loads(
            (SIMULATIONS / f"adversarial-{variant}" / "manifest.json").read_text(encoding="utf-8")
        )["dataset_id"]
        for variant in VARIANTS
    }
    assert len(set(ids.values())) == len(VARIANTS), ids
    assert set(ids.values()).isdisjoint(set(locked.values()))


# --------------------------------------------------------------------------- report shape


def test_the_report_records_that_nothing_was_retrained() -> None:
    report = build_report([], [ReferenceRow(label="x", source="y", available=False)])
    assert report["scoring"]["model_retrained"] is False
    assert report["scoring"]["threshold_changed"] is False
    assert report["cost_policy"]["false_negative_cost_units"] == FALSE_NEGATIVE_COST_UNITS
    assert report["cost_policy"]["false_positive_cost_units"] == FALSE_POSITIVE_COST_UNITS


def test_the_report_is_canonical_and_deterministic(tmp_path: Path) -> None:
    report = build_report([], [ReferenceRow(label="x", source="y", available=False)])
    first = write_report(report, tmp_path / "a").read_bytes()
    second = write_report(report, tmp_path / "b").read_bytes()
    assert first == second
    assert first.endswith(b"\n")
    # Canonical form, checked by round-tripping rather than by scanning for ", " -- that substring
    # legitimately appears inside the report's own prose.
    canonical = (
        json.dumps(json.loads(first), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    assert first == canonical


@report_only
def test_the_generated_report_carries_no_raw_identifier() -> None:
    import re

    text = REPORT.read_text(encoding="utf-8")
    assert re.search(r"(evt|mrc|chk|cus|dev|net|ses|pmt)_[0-9a-f]{32}", text) is None
    assert "cmp_" not in text


@report_only
def test_the_generated_report_includes_the_within_dataset_control() -> None:
    """Without the control, a low score cannot be attributed to the evasion."""

    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    for variant in payload["variants"]:
        assert variant["control_attack_count"] > 0
        assert variant["control_recall"] is not None
        # The control is ordinary traffic in the same file, so it must score far higher.
        assert variant["control_recall"] > variant["recall"]


@report_only
def test_the_generated_report_reads_both_reference_rows_from_source() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    labels = [row["label"] for row in payload["references"]]
    assert labels == ["Gate B2 held-out", "Gate C1 policy-validation"]
    for row in payload["references"]:
        assert row["source"].startswith("artifacts")


# --------------------------------------------------------------------------- scoring end to end


def _write_fixture(root: Path, variant: str, attack_features: dict, legit_features: dict) -> None:
    """A minimal dataset pair on disk: enough to drive score_variant without generating one."""

    from riskloom.features.schema import FEATURE_NAMES

    simulation = root / "simulations" / f"adversarial-{variant}"
    features = root / "features" / f"adversarial-{variant}"
    simulation.mkdir(parents=True)
    features.mkdir(parents=True)

    labels, rows = [], []
    for index in range(12):
        event_id = f"evt_{index:032x}"
        is_attack = index % 3 == 0
        split = "test" if index < 9 else "train"
        labels.append(
            {
                "event_id": event_id,
                "is_attack": is_attack,
                "split": split,
                "scenario_type": "card_testing_campaign" if is_attack else "normal",
                "campaign_id": f"cmp_{index % 2:032x}" if is_attack else None,
            }
        )
        source = attack_features if is_attack else legit_features
        rows.append(
            {
                "event_id": event_id,
                "features": {name: source.get(name, 0) for name in FEATURE_NAMES},
            }
        )

    (simulation / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    (simulation / "manifest.json").write_text(
        json.dumps({"dataset_id": "a" * 64}), encoding="utf-8"
    )
    (features / "features.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (features / "manifest.json").write_text(
        json.dumps({"feature_dataset_id": "b" * 64}), encoding="utf-8"
    )


@pytest.mark.skipif(
    not Path("artifacts/models/development/model.json").exists(),
    reason="locked model artifact is Git-ignored",
)
def test_score_variant_separates_test_measurement_from_the_control(tmp_path: Path) -> None:
    """The control must come from train and calibration, never from the measured split."""

    from riskloom.analysis.adversarial_stress import score_variant
    from riskloom.modeling.config import load_modeling_config

    _write_fixture(
        tmp_path,
        "slow-low",
        attack_features={"device_prior_attempt_count_60s": 40},
        legit_features={},
    )
    result = score_variant(
        "slow-low",
        tmp_path / "simulations" / "adversarial-slow-low",
        tmp_path / "features" / "adversarial-slow-low",
        Path("artifacts/models/development"),
        load_modeling_config(Path("configs/modeling/default.json")),
    )

    assert result.row_count == 9, "only the test split is measured"
    assert result.control_attack_count == 1, "train rows form the control"
    assert result.attack_count == 3
    assert result.true_positive + result.false_negative == result.attack_count
    assert result.cost_units == result.false_negative * 25 + result.false_positive * 1
    assert result.campaigns_total == 2
    assert result.simulation_dataset_id == "a" * 64
    assert result.feature_dataset_id == "b" * 64


def test_the_cli_runs_the_analysis_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    from riskloom.analysis import cli

    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"variants": [{"variant": name} for name in VARIANTS]}

    monkeypatch.setattr(cli.adversarial_stress, "run", fake_run)
    assert cli.main(["adversarial-stress", "--output-dir", "out"]) == 0
    assert captured["model_dir"] == Path("artifacts/models/development")
    assert captured["output_dir"] == Path("out")


def test_the_cli_requires_a_command() -> None:
    from riskloom.analysis import cli

    with pytest.raises(SystemExit):
        cli.main([])
