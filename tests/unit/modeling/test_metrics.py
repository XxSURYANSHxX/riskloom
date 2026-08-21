from typing import Any

import numpy as np

from riskloom.modeling.metrics import (
    campaign_metrics,
    complete_metrics,
    hard_negative_metrics,
    probability_metrics,
    reliability_metrics,
    select_threshold,
    threshold_metrics,
)


def test_probability_metrics_keep_ap_and_trapezoidal_pr_auc_distinct() -> None:
    targets = np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.8, 0.7, 0.6, 0.2])
    metrics = probability_metrics(targets, probabilities)
    assert metrics["average_precision"] != metrics["pr_auc_trapezoidal"]
    assert set(metrics) == {
        "average_precision",
        "brier_score",
        "log_loss",
        "pr_auc_trapezoidal",
        "roc_auc",
    }


def test_threshold_sweep_minimizes_cost_with_deterministic_ties() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold, metrics = select_threshold(targets, probabilities, 1, 25)
    assert threshold == 0.6
    assert metrics["cost_units"] == 0
    assert threshold_metrics(targets, probabilities, 1.0, 1, 25)["recall"] == 0.0


def _sweep(
    targets: list[int], probabilities: list[float], fp_cost: int, fn_cost: int
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Return the selected threshold, its metrics, and every evaluated candidate."""

    target_array = np.asarray(targets, dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    threshold, metrics = select_threshold(target_array, probability_array, fp_cost, fn_cost)
    evaluated = [
        threshold_metrics(target_array, probability_array, candidate, fp_cost, fn_cost)
        for candidate in sorted({0.0, 1.0, *probability_array.tolist()})
    ]
    return threshold, metrics, evaluated


def test_threshold_tie_break_prefers_lower_false_positive_at_equal_cost() -> None:
    threshold, metrics, evaluated = _sweep([0, 0, 1, 1], [0.9, 0.9, 0.1, 0.1], 1, 1)
    cheapest = [item for item in evaluated if item["cost_units"] == 2]
    assert len(cheapest) == 3
    assert sorted(item["false_positive"] for item in cheapest) == [0, 2, 2]
    assert threshold == 1.0
    assert metrics["false_positive"] == 0


def test_threshold_tie_break_prefers_higher_true_positive_at_equal_cost_and_fp() -> None:
    threshold, metrics, evaluated = _sweep([0, 1, 1], [0.1, 0.5, 0.9], 1, 0)
    tied = [item for item in evaluated if item["cost_units"] == 0 and item["false_positive"] == 0]
    assert len(tied) == 3
    assert sorted(item["true_positive"] for item in tied) == [0, 1, 2]
    assert threshold == 0.5
    assert metrics["true_positive"] == 2


def test_threshold_tie_break_precision_level_cannot_change_a_decision() -> None:
    """Precision is tp/(tp+fp), so tying cost, FP and TP forces precision to tie too."""

    for targets, probabilities, fp_cost, fn_cost in (
        ([0, 0, 1, 1], [0.9, 0.9, 0.1, 0.1], 1, 1),
        ([0, 1, 1], [0.1, 0.5, 0.9], 1, 0),
        ([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 1, 25),
    ):
        _, _, evaluated = _sweep(targets, probabilities, fp_cost, fn_cost)
        for left in evaluated:
            for right in evaluated:
                if (
                    left["cost_units"] == right["cost_units"]
                    and left["false_positive"] == right["false_positive"]
                    and left["true_positive"] == right["true_positive"]
                ):
                    assert left["precision"] == right["precision"]


def test_threshold_tie_break_prefers_higher_threshold_when_all_else_equal() -> None:
    threshold, metrics, evaluated = _sweep([0, 1, 1], [0.6, 0.6, 0.6], 1, 1)
    tied = [item for item in evaluated if item["cost_units"] == 1]
    assert len(tied) == 2
    assert {item["threshold"] for item in tied} == {0.0, 0.6}
    assert all(item["false_positive"] == 1 for item in tied)
    assert all(item["true_positive"] == 2 for item in tied)
    assert len({item["precision"] for item in tied}) == 1
    assert threshold == 0.6
    assert metrics["threshold"] == 0.6


def test_threshold_selection_tolerates_undefined_precision() -> None:
    threshold, metrics, evaluated = _sweep([0, 1, 1], [0.1, 0.5, 0.9], 1, 0)
    undefined = [item for item in evaluated if item["precision"] is None]
    assert len(undefined) == 1
    assert undefined[0]["threshold"] == 1.0
    assert undefined[0]["true_positive"] == 0 and undefined[0]["false_positive"] == 0
    assert undefined[0]["cost_units"] == metrics["cost_units"]
    assert undefined[0]["false_positive"] == metrics["false_positive"]
    assert threshold == 0.5
    assert metrics["precision"] == 1.0


def test_reliability_bins_retain_empty_bins_as_null() -> None:
    result = reliability_metrics(np.asarray([0, 1]), np.asarray([0.01, 0.99]), 10)
    assert len(result["bins"]) == 10
    assert result["bins"][1]["count"] == 0
    assert result["bins"][1]["attack_rate"] is None


def test_empty_hard_negative_slices_have_null_rates() -> None:
    result = hard_negative_metrics(
        np.asarray([0, 1]),
        np.asarray([0.2, 0.8]),
        0.5,
        ("normal", "card_testing_campaign"),
    )
    assert result["normal"]["false_positive_rate"] == 0.0
    assert result["flash_sale"] == {
        "false_positive_cost_units": 0,
        "false_positive_count": 0,
        "false_positive_rate": None,
        "false_positives_per_10000": None,
        "row_count": 0,
    }
    assert "card_testing_campaign" not in result


def test_missed_campaigns_contribute_zero_to_histogram() -> None:
    result = campaign_metrics(
        np.asarray([1, 1, 1, 1]),
        np.asarray([0.9, 0.8, 0.1, 0.2]),
        0.5,
        ("cmp_a", "cmp_a", "cmp_b", "cmp_b"),
        np.asarray([0, 100, 200, 300], dtype=np.int64),
    )
    assert result["missed_campaign_count"] == 1
    assert result["campaign_recall"] == 0.5
    assert result["detection_delay_ms"]["minimum"] == 0
    assert result["flagged_events_per_campaign"]["histogram"] == {"0": 1, "2": 1}


def test_complete_metrics_contains_aggregate_sections_only() -> None:
    result = complete_metrics(
        np.asarray([0, 1]),
        np.asarray([0.1, 0.9]),
        0.5,
        1,
        25,
        10,
        ("normal", "card_testing_campaign"),
        (None, "cmp_a"),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert set(result) == {
        "campaigns",
        "hard_negative_slices",
        "probability",
        "reliability",
        "threshold",
    }
