"""Read-only projection of the committed offline evaluation artifact.

The artifact lives under the Git-ignored ``artifacts/`` tree, so it is absent in a fresh clone.
Every function here treats absence as an ordinary outcome rather than an error: the endpoint
returns 404 and the dashboard renders an explicit "offline evaluation unavailable" state.

Nothing is recomputed. Every number is read from where it was already correctly computed and
written by Gate B2's ``evaluate-test`` run.
"""

import json
from pathlib import Path
from typing import Any

from riskloom.dashboard.schemas import (
    EvaluationCampaigns,
    EvaluationProbability,
    EvaluationReliability,
    EvaluationThreshold,
    HardNegativeSlice,
    ModelEvaluation,
    ReliabilityBin,
)


class EvaluationUnavailableError(RuntimeError):
    """The offline evaluation artifact is absent or unreadable."""


def _require(value: Any, key: str) -> Any:
    if not isinstance(value, dict) or key not in value:
        raise EvaluationUnavailableError("evaluation_artifact_malformed")
    return value[key]


def load_model_evaluation(path: Path) -> ModelEvaluation:
    """Project the evaluation artifact onto the dashboard's aggregate-only schema.

    Fields are selected by name. The file is never passed through, so a future artifact carrying a
    per-event section could not leak through this endpoint.
    """

    try:
        raw = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise EvaluationUnavailableError("evaluation_artifact_absent") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise EvaluationUnavailableError("evaluation_artifact_unreadable") from None

    if not isinstance(raw, dict):
        raise EvaluationUnavailableError("evaluation_artifact_malformed")

    metrics = _require(raw, "metrics")
    threshold = _require(metrics, "threshold")
    probability = _require(metrics, "probability")
    reliability = _require(metrics, "reliability")
    slices = _require(metrics, "hard_negative_slices")
    campaigns = _require(metrics, "campaigns")
    delay = _require(campaigns, "detection_delay_ms")
    flagged = _require(campaigns, "flagged_events_per_campaign")

    try:
        return ModelEvaluation(
            model_id=str(raw["model_id"]),
            evaluation_id=str(raw["evaluation_id"]),
            row_count=int(raw["row_count"]),
            threshold=EvaluationThreshold(
                row_count=int(threshold["row_count"]),
                attack_count=int(threshold["attack_count"]),
                legitimate_count=int(threshold["legitimate_count"]),
                true_positive=int(threshold["true_positive"]),
                false_positive=int(threshold["false_positive"]),
                true_negative=int(threshold["true_negative"]),
                false_negative=int(threshold["false_negative"]),
                precision=threshold["precision"],
                recall=threshold["recall"],
                specificity=threshold["specificity"],
                f1_score=threshold["f1_score"],
                false_positive_rate=threshold["false_positive_rate"],
                false_positives_per_10000_legitimate=threshold[
                    "false_positives_per_10000_legitimate"
                ],
                prevalence=threshold["prevalence"],
                review_count=int(threshold["review_count"]),
                review_rate=threshold["review_rate"],
                cost_units=int(threshold["cost_units"]),
                cost_per_10000_events=threshold["cost_per_10000_events"],
                # Rendered as a string for the same reason as every other probability here.
                threshold=repr(float(threshold["threshold"])),
            ),
            probability=EvaluationProbability(
                average_precision=float(probability["average_precision"]),
                pr_auc_trapezoidal=float(probability["pr_auc_trapezoidal"]),
                roc_auc=float(probability["roc_auc"]),
                brier_score=float(probability["brier_score"]),
                log_loss=float(probability["log_loss"]),
            ),
            reliability=EvaluationReliability(
                expected_calibration_error=float(reliability["expected_calibration_error"]),
                bins=[
                    ReliabilityBin(
                        lower_inclusive=float(entry["lower_inclusive"]),
                        upper_value=float(entry["upper_value"]),
                        upper_inclusive=bool(entry["upper_inclusive"]),
                        count=int(entry["count"]),
                        mean_probability=entry["mean_probability"],
                        attack_rate=entry["attack_rate"],
                    )
                    for entry in reliability["bins"]
                ],
            ),
            hard_negative_slices=[
                HardNegativeSlice(
                    slice_name=name,
                    row_count=int(entry["row_count"]),
                    false_positive_count=int(entry["false_positive_count"]),
                    false_positive_rate=entry["false_positive_rate"],
                    false_positives_per_10000=entry["false_positives_per_10000"],
                )
                for name, entry in sorted(slices.items())
            ],
            campaigns=EvaluationCampaigns(
                campaign_count=int(campaigns["campaign_count"]),
                detected_campaign_count=int(campaigns["detected_campaign_count"]),
                missed_campaign_count=int(campaigns["missed_campaign_count"]),
                campaign_recall=campaigns["campaign_recall"],
                detection_delay_ms_minimum=delay["minimum"],
                detection_delay_ms_median=delay["median"],
                detection_delay_ms_p95=delay["p95"],
                detection_delay_ms_maximum=delay["maximum"],
                flagged_per_campaign_minimum=flagged["minimum"],
                flagged_per_campaign_median=flagged["median"],
                flagged_per_campaign_maximum=flagged["maximum"],
            ),
        )
    except (KeyError, TypeError, ValueError):
        raise EvaluationUnavailableError("evaluation_artifact_malformed") from None
