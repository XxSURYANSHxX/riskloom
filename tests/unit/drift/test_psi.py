"""PSI arithmetic, asserted against independently recomputed values.

Every "exact" figure in this file was derived from the formula and cross-checked against a second,
differently-written implementation before being written down:

    PSI = sum over bins of (actual_share - expected_share) * ln(actual_share / expected_share)

The worked case, in full, so it can be rechecked by hand:

    reference [0.5, 0.5] vs actual [0.6, 0.4]
      bin 0:  (0.6 - 0.5) * ln(0.6 / 0.5) =  0.1 * ln(1.2) =  0.018232155679395
      bin 1:  (0.4 - 0.5) * ln(0.4 / 0.5) = -0.1 * ln(0.8) =  0.022314355131421
      PSI                                                   =  0.040546510810816

An earlier draft of this gate recorded 0.081093 for the same case by using a difference of 0.2 per
bin instead of 0.1. That is exactly twice the correct value, which is why the terms are spelled out
here rather than summarised.
"""

import math

import pytest

from riskloom.drift.psi import (
    BAND_MODERATE,
    BAND_SIGNIFICANT,
    PSI_EPSILON,
    PSI_MINIMUM_ROWS,
    band,
    contributions,
    from_counts,
    population_stability_index,
)

# Independently verified: see the module docstring for the full arithmetic.
TWO_BIN_60_40 = 0.040546510810816
TWO_BIN_70_30 = 0.169459572077441
TWO_BIN_80_20 = 0.415888308335967
FOUR_BIN_SHIFT = 0.109638511349342


def reference_implementation(expected: list[float], actual: list[float]) -> float:
    """A second formulation, expanded rather than factored.

    (a - e) * ln(a / e) == (a - e) * ln(a) - (a - e) * ln(e). Agreement between the two is a check
    on the implementation, not merely a restatement of it.
    """

    return sum(
        (a - e) * math.log(a) - (a - e) * math.log(e) for e, a in zip(expected, actual, strict=True)
    )


# --------------------------------------------------------------------------- exact values


def test_the_worked_two_bin_case_matches_the_hand_computation() -> None:
    term_zero = 0.1 * math.log(1.2)
    term_one = -0.1 * math.log(0.8)
    assert term_zero == pytest.approx(0.018232155679395, abs=1e-15)
    assert term_one == pytest.approx(0.022314355131421, abs=1e-15)
    assert term_zero + term_one == pytest.approx(TWO_BIN_60_40, abs=1e-15)
    assert population_stability_index([0.5, 0.5], [0.6, 0.4]) == pytest.approx(
        TWO_BIN_60_40, abs=1e-12
    )


def test_the_worked_case_is_not_the_doubled_value_an_earlier_draft_recorded() -> None:
    """Guards the specific error this gate had to correct."""

    computed = population_stability_index([0.5, 0.5], [0.6, 0.4])
    assert computed != pytest.approx(0.081093, abs=1e-5)
    assert 0.081093 / computed == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize(
    ("expected", "actual", "value"),
    [
        ([0.5, 0.5], [0.6, 0.4], TWO_BIN_60_40),
        ([0.5, 0.5], [0.7, 0.3], TWO_BIN_70_30),
        ([0.5, 0.5], [0.8, 0.2], TWO_BIN_80_20),
        ([0.4, 0.3, 0.2, 0.1], [0.25, 0.35, 0.25, 0.15], FOUR_BIN_SHIFT),
    ],
)
def test_exact_values_agree_with_an_independent_implementation(
    expected: list[float], actual: list[float], value: float
) -> None:
    assert population_stability_index(expected, actual) == pytest.approx(value, abs=1e-12)
    assert reference_implementation(expected, actual) == pytest.approx(value, abs=1e-12)


def test_identical_distributions_score_exactly_zero() -> None:
    for shares in ([0.5, 0.5], [0.25, 0.25, 0.5], [0.977824, 0.022176]):
        assert population_stability_index(shares, shares) == 0.0


def test_psi_is_symmetric() -> None:
    """Both the difference and the log ratio change sign, so the product does not."""

    expected = [0.4, 0.3, 0.2, 0.1]
    actual = [0.25, 0.35, 0.25, 0.15]
    assert population_stability_index(expected, actual) == pytest.approx(
        population_stability_index(actual, expected), abs=1e-15
    )


def test_contributions_sum_to_the_total() -> None:
    expected = [0.4, 0.3, 0.2, 0.1]
    actual = [0.25, 0.35, 0.25, 0.15]
    terms = contributions(expected, actual)
    assert len(terms) == 4
    assert sum(terms) == pytest.approx(population_stability_index(expected, actual), abs=1e-15)
    assert terms[0] == pytest.approx(0.070500544386860, abs=1e-12)


def test_every_contribution_is_non_negative() -> None:
    """Each term is (a-e)*ln(a/e); the factors share a sign, so no bin can subtract."""

    for term in contributions([0.4, 0.3, 0.2, 0.1], [0.25, 0.35, 0.25, 0.15]):
        assert term >= 0.0


def test_from_counts_normalises_before_scoring() -> None:
    value, terms = from_counts([50, 50], [600, 400])
    assert value == pytest.approx(TWO_BIN_60_40, abs=1e-12)
    assert len(terms) == 2


# --------------------------------------------------------------------------- bands


def test_band_boundaries_are_asserted_directly() -> None:
    """No tidy distribution lands on 0.1 or 0.25 exactly.

    Solving the symmetric two-bin family for those targets gives irrational proportions
    (0.6554746... and 0.7395492...), so constructing a fixture would test float arithmetic rather
    than the convention. The classifier is therefore asserted at the boundaries directly.
    """

    assert band(0.0) == "none"
    assert band(0.09999999) == "none"
    assert band(BAND_MODERATE) == "moderate"
    assert band(0.2) == "moderate"
    assert band(BAND_SIGNIFICANT) == "moderate"
    assert band(0.2500001) == "significant"
    assert band(1.0) == "significant"


def test_the_standard_convention_is_what_is_implemented() -> None:
    assert (BAND_MODERATE, BAND_SIGNIFICANT) == (0.1, 0.25)


@pytest.mark.parametrize(
    ("actual", "expected_band"),
    [
        ([0.6, 0.4], "none"),
        ([0.7, 0.3], "moderate"),
        ([0.8, 0.2], "significant"),
    ],
)
def test_end_to_end_band_for_verified_distributions(
    actual: list[float], expected_band: str
) -> None:
    assert band(population_stability_index([0.5, 0.5], actual)) == expected_band


# --------------------------------------------------------------------------- epsilon


def test_an_empty_reference_bin_yields_a_finite_value() -> None:
    """Five locked bins are empty; ln(x/0) would be undefined without the floor."""

    value = population_stability_index([0.5, 0.5, 0.0], [0.4, 0.4, 0.2])
    assert math.isfinite(value)
    assert value > 0


def test_the_epsilon_is_the_documented_constant() -> None:
    assert PSI_EPSILON == 1e-4


def test_epsilon_materially_changes_the_result() -> None:
    """The floor is load-bearing, and this records that rather than leaving it implicit.

    Same reference and same observation, four epsilons: the answer crosses a band boundary. A
    silent change to the constant therefore fails here rather than quietly moving every reading.
    """

    expected = [0.977824, 0.0, 0.0, 0.0, 0.000118, 0.0, 0.001647, 0.000882, 0.0, 0.019529]
    actual = [1.0] + [0.0] * 9

    values = {
        eps: population_stability_index(expected, actual, eps) for eps in (1e-3, 1e-4, 1e-5, 1e-6)
    }
    assert values[1e-3] == pytest.approx(0.0559, abs=5e-4)
    assert values[1e-4] == pytest.approx(0.1090, abs=5e-4)
    assert values[1e-5] == pytest.approx(0.1609, abs=5e-4)
    assert values[1e-6] == pytest.approx(0.2122, abs=5e-4)

    assert band(values[1e-3]) == "none"
    assert band(values[1e-4]) == "moderate"


def test_the_minimum_row_guard_is_the_documented_constant() -> None:
    assert PSI_MINIMUM_ROWS == 200


# --------------------------------------------------------------------------- input handling


def test_mismatched_bin_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="drift_bin_count_mismatch"):
        population_stability_index([0.5, 0.5], [1.0])


def test_an_empty_distribution_is_refused() -> None:
    with pytest.raises(ValueError, match="drift_empty_distribution"):
        from_counts([0, 0], [1, 1])
