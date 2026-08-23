"""Reference rows for the adversarial comparison, read from their source artifacts.

This module is the only place a reference figure enters the report, and it holds none of its own. A
hardcoded baseline is a number that quietly stops being true: if the held-out evaluation were ever
republished, a report carrying literals would keep printing the old figures and nothing would say
so. Everything here is read at run time from the artifact that produced it.

Because ``artifacts/`` is Git-ignored, either source may be absent. That is reported as
``reference_unavailable`` rather than filled in from memory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVALUATION_PATH = Path("artifacts/evaluations/development/evaluation.json")
POLICY_COMPARISON_PATH = Path("artifacts/policy/comparisons/development/comparison.json")


@dataclass(frozen=True)
class ReferenceRow:
    """One comparison row. ``available`` false means the source artifact was not present."""

    label: str
    source: str
    available: bool
    row_count: int | None = None
    attack_count: int | None = None
    recall: float | None = None
    precision: float | None = None
    average_precision: float | None = None
    false_positive_rate: float | None = None
    cost_units: int | None = None
    note: str = ""


class ReferenceError(ValueError):
    """A reference artifact exists but does not say what it must."""


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_held_out_reference(path: Path = EVALUATION_PATH) -> ReferenceRow:
    """Gate B2's one-time held-out evaluation.

    Read-only. The evaluation artifact is never reopened for scoring here -- only its already
    published aggregate metrics are read, which is not the ``evaluate-test`` path.
    """

    payload = _read(path)
    if payload is None:
        return ReferenceRow(
            label="Gate B2 held-out",
            source=str(path),
            available=False,
            note="locked evaluation artifact not present in this checkout",
        )

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ReferenceError("held_out_reference_missing_metrics")
    threshold = metrics.get("threshold")
    probability = metrics.get("probability")
    if not isinstance(threshold, dict) or not isinstance(probability, dict):
        raise ReferenceError("held_out_reference_missing_sections")

    try:
        return ReferenceRow(
            label="Gate B2 held-out",
            source=str(path),
            available=True,
            row_count=int(threshold["row_count"]),
            attack_count=int(threshold["attack_count"]),
            recall=float(threshold["recall"]),
            precision=float(threshold["precision"]),
            average_precision=float(probability["average_precision"]),
            false_positive_rate=float(threshold["false_positive_rate"]),
            cost_units=int(threshold["cost_units"]),
            note="real held-out partition, scored once",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceError("held_out_reference_fields_invalid") from exc


def load_policy_validation_reference(path: Path = POLICY_COMPARISON_PATH) -> ReferenceRow:
    """Gate C1's incumbent policy on the counterfactual validation batch.

    Recall and precision are derived from the stored confusion counts rather than read, because the
    artifact records counts rather than rates. The derived false-positive rate is then checked
    against the rate the artifact stores itself, so a doctored file whose counts contradict its own
    summary is rejected instead of quietly producing a plausible row.

    Average precision is genuinely absent from this artifact and is reported as ``None``. The 0.4384
    quoted in the README came from a separate blindspot measurement; copying it here would be
    exactly the hardcoding this module exists to prevent.
    """

    payload = _read(path)
    if payload is None:
        return ReferenceRow(
            label="Gate C1 policy-validation",
            source=str(path),
            available=False,
            note="policy comparison artifact not present in this checkout",
        )

    incumbent = payload.get("incumbent_policy")
    outcome = incumbent.get("outcome") if isinstance(incumbent, dict) else None
    if not isinstance(outcome, dict):
        raise ReferenceError("policy_reference_missing_incumbent_outcome")

    try:
        true_positive = int(outcome["true_positive"])
        false_negative = int(outcome["false_negative"])
        false_positive = int(outcome["false_positive"])
        legitimate_count = int(outcome["legitimate_count"])
        attack_count = int(outcome["attack_count"])
        row_count = int(outcome["row_count"])
        cost_units = int(outcome["cost_units"])
        stored_rate = float(outcome["false_positive_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceError("policy_reference_fields_invalid") from exc

    if attack_count <= 0 or legitimate_count <= 0:
        raise ReferenceError("policy_reference_counts_invalid")
    if true_positive + false_negative != attack_count:
        raise ReferenceError("policy_reference_attack_counts_inconsistent")

    derived_rate = false_positive / legitimate_count
    if abs(derived_rate - stored_rate) > 1e-9:
        raise ReferenceError("policy_reference_false_positive_rate_inconsistent")

    flagged = true_positive + false_positive
    return ReferenceRow(
        label="Gate C1 policy-validation",
        source=str(path),
        available=True,
        row_count=row_count,
        attack_count=attack_count,
        recall=true_positive / attack_count,
        precision=(true_positive / flagged) if flagged else None,
        average_precision=None,
        false_positive_rate=stored_rate,
        cost_units=cost_units,
        note="counterfactual batch; average precision is not recorded in this artifact",
    )


def load_references() -> list[ReferenceRow]:
    return [load_held_out_reference(), load_policy_validation_reference()]
