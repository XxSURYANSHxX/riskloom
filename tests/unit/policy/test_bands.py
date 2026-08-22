import itertools

import numpy as np
import pytest
from pydantic import ValidationError

from riskloom.policy.bands import (
    ALLOW_CODE,
    DENY_CODE,
    REVIEW_CODE,
    TIE_BREAK_ORDER,
    BandPolicy,
    CostPolicy,
    PolicyBandError,
    band_decisions,
    boundary_diagnostics,
    evaluate_band,
    evaluate_single_threshold,
    select_band,
    sweep_candidates,
)

COSTS = CostPolicy(false_negative_cost_units=25, false_positive_cost_units=1, review_cost_units=3)


def _brute_force(
    targets: np.ndarray, probabilities: np.ndarray, costs: CostPolicy
) -> tuple[int, set[tuple[float, float]]]:
    candidates = sorted(set(probabilities.tolist()) | {0.0, 1.0})
    best = None
    winners: set[tuple[float, float]] = set()
    for low, high in itertools.product(candidates, repeat=2):
        if low > high:
            continue
        band = BandPolicy(lower_threshold=low, upper_threshold=high)
        cost = evaluate_band(targets, probabilities, band, costs).cost_units
        if best is None or cost < best:
            best = cost
            winners = {(low, high)}
        elif cost == best:
            winners.add((low, high))
    assert best is not None
    return best, winners


def test_band_routes_by_probability_alone() -> None:
    band = BandPolicy(lower_threshold=0.2, upper_threshold=0.8)
    probabilities = np.asarray([0.0, 0.19, 0.2, 0.5, 0.79, 0.8, 1.0])
    assert band_decisions(probabilities, band).tolist() == [
        ALLOW_CODE,
        ALLOW_CODE,
        REVIEW_CODE,
        REVIEW_CODE,
        REVIEW_CODE,
        DENY_CODE,
        DENY_CODE,
    ]


def test_equal_thresholds_reduce_to_the_incumbent_single_threshold_policy() -> None:
    targets = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray([0.1, 0.9, 0.4, 0.6])
    band = BandPolicy(lower_threshold=0.5, upper_threshold=0.5)
    assert not band.has_review_tier
    banded = evaluate_band(targets, probabilities, band, COSTS)
    single = evaluate_single_threshold(targets, probabilities, 0.5, COSTS)
    assert banded == single
    assert banded.review_count == 0


def test_review_is_neither_a_false_positive_nor_a_false_negative() -> None:
    targets = np.asarray([1, 0])
    probabilities = np.asarray([0.5, 0.5])
    band = BandPolicy(lower_threshold=0.4, upper_threshold=0.9)
    outcome = evaluate_band(targets, probabilities, band, COSTS)
    assert outcome.review_count == 2
    assert outcome.false_negative == 0
    assert outcome.false_positive == 0
    assert outcome.true_positive == 0
    # Review cost is charged for both rows regardless of their true labels.
    assert outcome.cost_units == 2 * COSTS.review_cost_units


def test_cost_is_the_three_term_function() -> None:
    targets = np.asarray([1, 1, 0, 0, 0])
    probabilities = np.asarray([0.01, 0.5, 0.5, 0.99, 0.02])
    band = BandPolicy(lower_threshold=0.1, upper_threshold=0.9)
    outcome = evaluate_band(targets, probabilities, band, COSTS)
    assert (outcome.false_negative, outcome.false_positive, outcome.review_count) == (1, 1, 2)
    assert outcome.cost_units == 1 * 25 + 1 * 1 + 2 * 3


# A middle band holding 8 attacks and 12 legitimate rows, flanked by clean allow and deny regions.
_MIDDLE_BAND_TARGETS = np.asarray([0] * 40 + [1] * 8 + [0] * 12 + [1] * 10)
_MIDDLE_BAND_PROBABILITIES = np.asarray([0.01] * 40 + [0.5] * 20 + [0.99] * 10)


def test_a_review_tier_is_selected_when_review_is_cheaper_than_denying() -> None:
    """The sweep must be able to produce a genuine three-tier policy, not only degenerate ones."""

    costs = CostPolicy(
        false_negative_cost_units=25, false_positive_cost_units=10, review_cost_units=3
    )
    band, outcome = select_band(_MIDDLE_BAND_TARGETS, _MIDDLE_BAND_PROBABILITIES, costs, 64)
    assert band.has_review_tier
    assert outcome.review_count == 20
    assert outcome.false_negative == 0
    assert outcome.false_positive == 0
    minimum, _ = _brute_force(_MIDDLE_BAND_TARGETS, _MIDDLE_BAND_PROBABILITIES, costs)
    assert outcome.cost_units == minimum


def test_review_tier_is_unreachable_when_review_costs_at_least_a_false_positive() -> None:
    """A structural property of the cost function, not an accident of one dataset.

    Per row, DENY costs ``false_positive_cost_units`` if the row is legitimate and nothing if it is
    an attack, so DENY never costs more than ``false_positive_cost_units``. REVIEW always costs
    ``review_cost_units``. When ``review_cost_units >= false_positive_cost_units`` there is
    therefore always an optimal solution with an empty review tier, whatever the data looks like.
    The inherited Day 4 scale has a false positive at 1 unit, so any integer review cost of 1 or
    more makes the review tier unreachable by construction.
    """

    for review_cost in (1, 3, 10):
        costs = CostPolicy(
            false_negative_cost_units=25,
            false_positive_cost_units=1,
            review_cost_units=review_cost,
        )
        band, outcome = select_band(_MIDDLE_BAND_TARGETS, _MIDDLE_BAND_PROBABILITIES, costs, 64)
        assert not band.has_review_tier, review_cost
        assert outcome.review_count == 0, review_cost


def test_a_review_tier_is_rejected_when_the_middle_band_is_mostly_legitimate() -> None:
    costs = CostPolicy(
        false_negative_cost_units=25, false_positive_cost_units=10, review_cost_units=3
    )
    targets = np.asarray([0] * 40 + [1] * 1 + [0] * 19 + [1] * 10)
    probabilities = np.asarray([0.01] * 40 + [0.5] * 20 + [0.99] * 10)
    band, outcome = select_band(targets, probabilities, costs, 64)
    assert not band.has_review_tier
    assert outcome.review_count == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sweep_matches_exhaustive_brute_force(seed: int) -> None:
    generator = np.random.default_rng(seed)
    targets = (generator.random(60) < 0.3).astype(np.int64)
    probabilities = np.round(generator.random(60), 2)
    band, outcome = select_band(targets, probabilities, COSTS, 4_096)
    minimum, winners = _brute_force(targets, probabilities, COSTS)
    assert outcome.cost_units == minimum
    assert (band.lower_threshold, band.upper_threshold) in winners


def test_sweep_is_deterministic_across_row_permutations() -> None:
    generator = np.random.default_rng(11)
    targets = (generator.random(80) < 0.25).astype(np.int64)
    probabilities = np.round(generator.random(80), 3)
    first, _ = select_band(targets, probabilities, COSTS, 512)
    order = generator.permutation(targets.shape[0])
    second, _ = select_band(targets[order], probabilities[order], COSTS, 512)
    assert first == second


def test_incumbent_threshold_is_always_a_candidate() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.3])
    candidates = sweep_candidates(probabilities, 4, required=(0.0033862949155182734,))
    assert 0.0033862949155182734 in candidates
    assert candidates == tuple(sorted(candidates))


def test_fitted_band_never_loses_to_the_incumbent_on_the_fit_partition() -> None:
    generator = np.random.default_rng(5)
    targets = (generator.random(400) < 0.02).astype(np.int64)
    probabilities = np.clip(generator.random(400) ** 4, 1e-15, 1 - 1e-15)
    incumbent_threshold = 0.25
    _, outcome = select_band(targets, probabilities, COSTS, 256, required=(incumbent_threshold,))
    incumbent = evaluate_single_threshold(targets, probabilities, incumbent_threshold, COSTS)
    assert outcome.cost_units <= incumbent.cost_units


def test_band_rejects_inverted_thresholds_and_invalid_vectors() -> None:
    with pytest.raises(ValidationError):
        BandPolicy(lower_threshold=0.9, upper_threshold=0.1)
    with pytest.raises(ValidationError):
        BandPolicy(lower_threshold=-0.1, upper_threshold=0.5)
    band = BandPolicy(lower_threshold=0.1, upper_threshold=0.5)
    with pytest.raises(PolicyBandError, match="probability_vector_invalid"):
        band_decisions(np.asarray([0.5, np.nan]), band)
    with pytest.raises(PolicyBandError, match="target_vector_invalid"):
        evaluate_band(np.asarray([0, 2]), np.asarray([0.1, 0.2]), band, COSTS)
    with pytest.raises(PolicyBandError, match="shape_invalid"):
        evaluate_band(np.asarray([0, 1, 0]), np.asarray([0.1, 0.2]), band, COSTS)
    with pytest.raises(PolicyBandError, match="requires_rows"):
        select_band(np.asarray([], dtype=np.int64), np.asarray([]), COSTS, 16)


def test_tie_break_order_is_documented_and_omits_precision() -> None:
    assert TIE_BREAK_ORDER == (
        "minimum_cost",
        "minimum_false_positive",
        "maximum_true_positive",
        "minimum_review_count",
        "maximum_upper_threshold",
        "maximum_lower_threshold",
        "stable_candidate_index",
    )
    assert "precision" not in " ".join(TIE_BREAK_ORDER)


def test_tie_break_prefers_fewer_reviews_then_higher_thresholds_at_equal_cost() -> None:
    # Every row is legitimate and scores below 0.5, so any band with both thresholds at or above
    # 0.5 costs zero. The ladder must take the one with no reviews and the highest thresholds.
    targets = np.asarray([0, 0, 0, 1])
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.9])
    band, outcome = select_band(targets, probabilities, COSTS, 64)
    assert outcome.cost_units == 0
    assert outcome.review_count == 0
    assert band.lower_threshold == band.upper_threshold == 0.9


def test_basis_point_reporting_is_integer_and_null_safe() -> None:
    targets = np.asarray([1, 1])
    probabilities = np.asarray([0.9, 0.95])
    band = BandPolicy(lower_threshold=0.5, upper_threshold=0.5)
    report = evaluate_band(targets, probabilities, band, COSTS).as_report()
    assert report["legitimate_count"] == 0
    assert report["false_positive_rate"] is None
    assert report["false_positive_rate_basis_points"] is None
    assert report["false_positives_per_10000_legitimate"] is None


def test_boundary_diagnostics_expose_tie_clusters_below_a_threshold() -> None:
    # Eight rows share one value; the threshold sits just above it and is not itself observed.
    probabilities = np.asarray([0.1] * 8 + [0.9] * 2)
    diagnostics = boundary_diagnostics(probabilities, 0.5)
    assert diagnostics["threshold_is_an_observed_value"] is False
    assert diagnostics["nearest_observed_value_below_threshold"] == 0.1
    assert diagnostics["tied_rows_at_nearest_value_below_threshold"] == 8
    assert diagnostics["distinct_probability_count"] == 2


def test_boundary_diagnostics_report_an_observed_threshold_as_stable() -> None:
    probabilities = np.asarray([0.1, 0.5, 0.5, 0.9])
    diagnostics = boundary_diagnostics(probabilities, 0.5)
    assert diagnostics["threshold_is_an_observed_value"] is True
    assert diagnostics["nearest_observed_value_below_threshold"] == 0.1
    assert diagnostics["tied_rows_at_nearest_value_below_threshold"] == 1


def test_boundary_diagnostics_handle_a_threshold_below_everything() -> None:
    diagnostics = boundary_diagnostics(np.asarray([0.4, 0.6]), 0.0)
    assert diagnostics["nearest_observed_value_below_threshold"] is None
    assert diagnostics["tied_rows_at_nearest_value_below_threshold"] == 0


def test_boundary_diagnostics_reject_invalid_vectors() -> None:
    with pytest.raises(PolicyBandError, match="probability_vector_invalid"):
        boundary_diagnostics(np.asarray([0.1, np.nan]), 0.5)
