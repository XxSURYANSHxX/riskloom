from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _confusion(targets: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    return {
        "false_negative": int(np.count_nonzero((targets == 1) & ~flagged)),
        "false_positive": int(np.count_nonzero((targets == 0) & flagged)),
        "true_negative": int(np.count_nonzero((targets == 0) & ~flagged)),
        "true_positive": int(np.count_nonzero((targets == 1) & flagged)),
    }


def threshold_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: int,
    false_negative_cost: int,
) -> dict[str, Any]:
    flagged = probabilities >= threshold
    counts = _confusion(targets, flagged)
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    positive = counts["true_positive"] + counts["false_negative"]
    legitimate = counts["true_negative"] + counts["false_positive"]
    precision = _rate(counts["true_positive"], predicted_positive)
    recall = _rate(counts["true_positive"], positive)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    total = int(targets.shape[0])
    cost_units = (
        counts["false_positive"] * false_positive_cost
        + counts["false_negative"] * false_negative_cost
    )
    return {
        **counts,
        "attack_count": positive,
        "cost_per_10000_events": _rate(cost_units * 10_000, total),
        "cost_units": cost_units,
        "f1_score": f1_score,
        "false_positives_per_10000_legitimate": _rate(
            counts["false_positive"] * 10_000, legitimate
        ),
        "false_positive_rate": _rate(counts["false_positive"], legitimate),
        "legitimate_count": legitimate,
        "precision": precision,
        "prevalence": _rate(positive, total),
        "recall": recall,
        "review_count": predicted_positive,
        "review_rate": _rate(predicted_positive, total),
        "row_count": total,
        "specificity": _rate(counts["true_negative"], legitimate),
        "threshold": threshold,
    }


def select_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    false_positive_cost: int,
    false_negative_cost: int,
) -> tuple[float, dict[str, Any]]:
    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    evaluated = [
        threshold_metrics(
            targets, probabilities, threshold, false_positive_cost, false_negative_cost
        )
        for threshold in candidates
    ]

    def key(metrics: dict[str, Any]) -> tuple[float, int, int, float, float]:
        precision = metrics["precision"]
        return (
            float(metrics["cost_units"]),
            int(metrics["false_positive"]),
            -int(metrics["true_positive"]),
            -float(precision) if precision is not None else 0.0,
            -float(metrics["threshold"]),
        )

    selected = min(evaluated, key=key)
    return float(selected["threshold"]), selected


def probability_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    return {
        "average_precision": float(average_precision_score(targets, probabilities)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
    }


def reliability_metrics(
    targets: np.ndarray, probabilities: np.ndarray, bin_count: int
) -> dict[str, Any]:
    bins: list[dict[str, float | int | None]] = []
    ece = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bin_count - 1 else probabilities < upper
        )
        count = int(np.count_nonzero(selected))
        mean_probability = float(np.mean(probabilities[selected])) if count else None
        attack_rate = float(np.mean(targets[selected])) if count else None
        if count and mean_probability is not None and attack_rate is not None:
            ece += count / targets.shape[0] * abs(mean_probability - attack_rate)
        bins.append(
            {
                "attack_rate": attack_rate,
                "count": count,
                "lower_inclusive": lower,
                "mean_probability": mean_probability,
                "upper_inclusive": index == bin_count - 1,
                "upper_value": upper,
            }
        )
    return {"bins": bins, "expected_calibration_error": ece}


def hard_negative_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    scenarios: tuple[str, ...],
) -> dict[str, dict[str, float | int | None]]:
    flagged = probabilities >= threshold
    result: dict[str, dict[str, float | int | None]] = {}
    legitimate_scenarios = (
        "flash_sale",
        "legitimate_failure",
        "legitimate_retry",
        "normal",
        "shared_infrastructure",
    )
    for scenario in legitimate_scenarios:
        mask = np.asarray(
            [
                target == 0 and value == scenario
                for target, value in zip(targets, scenarios, strict=True)
            ]
        )
        count = int(np.count_nonzero(mask))
        flagged_count = int(np.count_nonzero(flagged & mask))
        result[scenario] = {
            "false_positive_cost_units": flagged_count,
            "false_positive_count": flagged_count,
            "false_positive_rate": _rate(flagged_count, count),
            "false_positives_per_10000": _rate(flagged_count * 10_000, count),
            "row_count": count,
        }
    return result


def campaign_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    campaign_ids: tuple[str | None, ...],
    occurred_at_ms: np.ndarray,
) -> dict[str, Any]:
    flagged = probabilities >= threshold
    counts: dict[str, int] = {}
    total: dict[str, int] = {}
    first_attack_ms: dict[str, int] = {}
    first_flagged_ms: dict[str, int] = {}
    for index, (target, campaign_id) in enumerate(zip(targets, campaign_ids, strict=True)):
        if target != 1 or campaign_id is None:
            continue
        total[campaign_id] = total.get(campaign_id, 0) + 1
        counts.setdefault(campaign_id, 0)
        first_attack_ms.setdefault(campaign_id, int(occurred_at_ms[index]))
        if flagged[index]:
            counts[campaign_id] += 1
            first_flagged_ms.setdefault(campaign_id, int(occurred_at_ms[index]))
    values = sorted(counts.values())
    histogram = Counter(values)
    detected_delays = sorted(
        first_flagged_ms[campaign] - first_attack_ms[campaign] for campaign in first_flagged_ms
    )

    def percentile(numbers: list[int], numerator: int) -> int | None:
        if not numbers:
            return None
        index = (len(numbers) * numerator + 99) // 100 - 1
        return numbers[max(index, 0)]

    return {
        "campaign_count": len(values),
        "campaign_recall": _rate(len(first_flagged_ms), len(values)),
        "detected_campaign_count": len(first_flagged_ms),
        "detection_delay_ms": {
            "maximum": max(detected_delays) if detected_delays else None,
            "median": percentile(detected_delays, 50),
            "minimum": min(detected_delays) if detected_delays else None,
            "p95": percentile(detected_delays, 95),
        },
        "flagged_events_per_campaign": {
            "histogram": {str(key): histogram[key] for key in sorted(histogram)},
            "maximum": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "median": percentile(values, 50),
            "minimum": min(values) if values else None,
            "p95": percentile(values, 95),
        },
        "missed_campaign_count": len(values) - len(first_flagged_ms),
    }


def complete_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: int,
    false_negative_cost: int,
    bin_count: int,
    scenarios: tuple[str, ...],
    campaign_ids: tuple[str | None, ...],
    occurred_at_ms: np.ndarray,
) -> dict[str, Any]:
    return {
        "campaigns": campaign_metrics(
            targets, probabilities, threshold, campaign_ids, occurred_at_ms
        ),
        "hard_negative_slices": hard_negative_metrics(targets, probabilities, threshold, scenarios),
        "probability": probability_metrics(targets, probabilities),
        "reliability": reliability_metrics(targets, probabilities, bin_count),
        "threshold": threshold_metrics(
            targets,
            probabilities,
            threshold,
            false_positive_cost,
            false_negative_cost,
        ),
    }
