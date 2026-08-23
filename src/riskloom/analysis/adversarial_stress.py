"""Score the locked Day 4 model against evasion-shaped attack traffic.

Pure offline analysis, in the same shape as :mod:`riskloom.analysis.blindspot`: invoked by command,
producing a written report, reachable from no decision path and reaching none.

Nothing here trains, re-locks, re-thresholds, or otherwise touches the locked model. It loads the
published artifact, runs portable JSON inference, and compares the outcome against two reference
rows read from their own source artifacts. Results are reported as measured. A variant that fools
the model is a finding, not a failure of the run.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from riskloom.analysis.references import ReferenceRow, load_references
from riskloom.features.schema import FEATURE_NAMES
from riskloom.modeling.artifacts import load_locked_model
from riskloom.modeling.config import ModelingConfig, load_modeling_config
from riskloom.modeling.model import portable_probabilities

ANALYSIS_SCHEMA_VERSION = "1.0.0"

# Inherited unchanged from the locked Day 4 configuration so every row in the comparison is scored
# by the same rule. These are abstract policy units, never a currency claim.
FALSE_NEGATIVE_COST_UNITS = 25
FALSE_POSITIVE_COST_UNITS = 1

VARIANTS = ("slow-low", "window-edge", "distributed", "failure-camouflage")

VARIANT_TARGETS = {
    "slow-low": "velocity counts in the 60s and 300s windows",
    "window-edge": "the left-exclusive (t - window, t] boundary rule",
    "distributed": "device and network reuse concentration",
    "failure-camouflage": "the 18 failure-derived features",
}


@dataclass(frozen=True)
class VariantResult:
    variant: str
    targets: str
    simulation_dataset_id: str
    feature_dataset_id: str
    row_count: int
    attack_count: int
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    recall: float | None
    precision: float | None
    average_precision: float | None
    false_positive_rate: float | None
    cost_units: int
    campaigns_total: int
    campaigns_detected: int
    mean_attack_probability: float
    mean_legitimate_probability: float
    # Within-dataset control: the same file's train and calibration splits carry no evasion at all,
    # so they answer the question a cross-dataset comparison cannot -- whether a low score means the
    # evasion worked or merely that this dataset is shaped differently from the held-out one.
    control_attack_count: int
    control_recall: float | None
    control_mean_attack_probability: float


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Area under the precision-recall curve by the step-wise definition.

    Deliberately not the trapezoidal PR-AUC: the project keeps those two distinct, and this is the
    same quantity the held-out evaluation reports, so the comparison stays like-for-like.
    """

    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    cumulative_true = np.cumsum(ranked)
    precision_at_k = cumulative_true / np.arange(1, len(ranked) + 1)
    return float((precision_at_k * ranked).sum() / labels.sum())


def score_variant(
    variant: str,
    simulation_dir: Path,
    features_dir: Path,
    model_dir: Path,
    modeling_config: ModelingConfig,
) -> VariantResult:
    """Score one variant's test-split attacks against the locked model."""

    model, _report, _manifest = load_locked_model(model_dir, modeling_config)
    threshold = model.decision_threshold

    labels_by_event = {row["event_id"]: row for row in _load_jsonl(simulation_dir / "labels.jsonl")}
    feature_rows = _load_jsonl(features_dir / "features.jsonl")

    matrix: list[list[int]] = []
    truth: list[int] = []
    campaigns: list[str | None] = []
    control_matrix: list[list[int]] = []
    control_truth: list[int] = []
    for row in feature_rows:
        label = labels_by_event[row["event_id"]]
        values = row.get("features", row)
        vector = [int(values[name]) for name in FEATURE_NAMES]
        split = label.get("split")
        # The evasion shape is applied to the test split only, so that is where the measurement is
        # taken. Train and calibration are ordinary traffic and form the control.
        if split == "test":
            matrix.append(vector)
            truth.append(1 if label.get("is_attack") else 0)
            campaigns.append(label.get("campaign_id"))
        elif split in ("train", "calibration"):
            control_matrix.append(vector)
            control_truth.append(1 if label.get("is_attack") else 0)

    features = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    probabilities = portable_probabilities(model, features)
    flagged = probabilities >= threshold

    control_features = np.asarray(control_matrix, dtype=np.float64)
    control_labels = np.asarray(control_truth, dtype=np.int64)
    control_probabilities = portable_probabilities(model, control_features)
    control_attacks = int(control_labels.sum())
    control_hits = int(np.sum((control_probabilities >= threshold) & (control_labels == 1)))

    true_positive = int(np.sum(flagged & (labels == 1)))
    false_negative = int(np.sum(~flagged & (labels == 1)))
    false_positive = int(np.sum(flagged & (labels == 0)))
    true_negative = int(np.sum(~flagged & (labels == 0)))
    attack_count = int(labels.sum())
    legitimate_count = int(len(labels) - attack_count)

    detected: set[str] = set()
    total: set[str] = set()
    for campaign, is_attack, hit in zip(campaigns, labels, flagged, strict=True):
        if is_attack and campaign is not None:
            total.add(campaign)
            if hit:
                detected.add(campaign)

    return VariantResult(
        variant=variant,
        targets=VARIANT_TARGETS[variant],
        simulation_dataset_id=json.loads(
            (simulation_dir / "manifest.json").read_text(encoding="utf-8")
        )["dataset_id"],
        feature_dataset_id=json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))[
            "feature_dataset_id"
        ],
        row_count=int(len(labels)),
        attack_count=attack_count,
        true_positive=true_positive,
        false_negative=false_negative,
        false_positive=false_positive,
        true_negative=true_negative,
        recall=(true_positive / attack_count) if attack_count else None,
        precision=(
            (true_positive / (true_positive + false_positive))
            if (true_positive + false_positive)
            else None
        ),
        average_precision=average_precision(labels, probabilities),
        false_positive_rate=(false_positive / legitimate_count) if legitimate_count else None,
        cost_units=false_negative * FALSE_NEGATIVE_COST_UNITS
        + false_positive * FALSE_POSITIVE_COST_UNITS,
        campaigns_total=len(total),
        campaigns_detected=len(detected),
        mean_attack_probability=float(probabilities[labels == 1].mean()) if attack_count else 0.0,
        mean_legitimate_probability=(
            float(probabilities[labels == 0].mean()) if legitimate_count else 0.0
        ),
        control_attack_count=control_attacks,
        control_recall=(control_hits / control_attacks) if control_attacks else None,
        control_mean_attack_probability=(
            float(control_probabilities[control_labels == 1].mean()) if control_attacks else 0.0
        ),
    )


def build_report(results: list[VariantResult], references: list[ReferenceRow]) -> dict[str, Any]:
    return {
        "product": "RiskLoom",
        "artifact_type": "adversarial_stress_report",
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "cost_policy": {
            "false_negative_cost_units": FALSE_NEGATIVE_COST_UNITS,
            "false_positive_cost_units": FALSE_POSITIVE_COST_UNITS,
            "interpretation": "abstract_policy_units_not_currency",
        },
        "scoring": {
            "inference": "portable_json",
            "model_retrained": False,
            "threshold_changed": False,
            "measured_on": "test split only, where the evasion shape is applied",
            "control": (
                "train and calibration splits of the same dataset carry no evasion, so they "
                "isolate the evasion effect from any difference in dataset shape or scale"
            ),
        },
        "references": [asdict(row) for row in references],
        "variants": [asdict(result) for result in results],
        "interpretation": (
            "Synthetic evasion traffic, not a held-out measurement. Recall on roughly 120 attacks "
            "carries about four percentage points of sampling noise, so read large movements and "
            "ignore small ones."
        ),
    }


def write_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "adversarial_stress.json"
    path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def run(
    simulations_root: Path,
    features_root: Path,
    model_dir: Path,
    output_dir: Path,
    modeling_config_path: Path,
) -> dict[str, Any]:
    modeling_config = load_modeling_config(modeling_config_path)
    results = [
        score_variant(
            variant,
            simulations_root / f"adversarial-{variant}",
            features_root / f"adversarial-{variant}",
            model_dir,
            modeling_config,
        )
        for variant in VARIANTS
    ]
    report = build_report(results, load_references())
    write_report(report, output_dir)
    return report
