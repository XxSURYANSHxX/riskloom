"""Population Stability Index over the calibrated probability score.

    PSI = sum over bins of (actual_share - expected_share) * ln(actual_share / expected_share)

Two properties worth stating because they make the tests meaningful: PSI is zero exactly when the
two distributions are identical, and it is symmetric -- swapping expected and actual gives the same
value, because both the difference and the log ratio change sign together.

**The epsilon is load-bearing, not a detail.** The locked reference has five empty bins, and
``ln(x/0)`` is undefined, so empty shares are floored at ``PSI_EPSILON``. That floor materially
moves the answer. Measured against the real 13-row ledger at the time of writing:

    epsilon   PSI      band
    1e-3      0.0559   no shift
    1e-4      0.1090   moderate      <- the value this module uses
    1e-5      0.1609   moderate
    1e-6      0.2122   moderate

A single constant therefore decides whether the same data reads as "no shift" or "moderate". It is
pinned here, asserted in the tests, and must not be changed without re-recording that table.
"""

import math
from typing import Literal

PSI_EPSILON = 1e-4
"""Floor applied to a zero share. See the module docstring: this value changes the band."""

PSI_MINIMUM_ROWS = 200
"""Below this many scored rows, no PSI is reported at all.

PSI over a handful of rows is noise wearing a decimal point. The real ledger currently holds 13
scored decisions, and emitting a band from that would be the most misleading number on the screen,
so the surface reports ``insufficient_data`` instead of a number it cannot support.
"""

BAND_MODERATE = 0.1
BAND_SIGNIFICANT = 0.25

DriftBand = Literal["none", "moderate", "significant"]


def band(value: float) -> DriftBand:
    """The standard convention: <0.1 none, 0.1-0.25 moderate, >0.25 significant.

    The boundaries are inclusive at the lower edge of each band, so exactly 0.1 and exactly 0.25
    both read as ``moderate``. That is the published convention and is asserted directly rather
    than inferred from a constructed distribution, because no distribution with tidy proportions
    lands on either boundary exactly.
    """

    if value < BAND_MODERATE:
        return "none"
    if value <= BAND_SIGNIFICANT:
        return "moderate"
    return "significant"


def _shares(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        raise ValueError("drift_empty_distribution")
    return [count / total for count in counts]


def contributions(
    expected: list[float], actual: list[float], epsilon: float = PSI_EPSILON
) -> list[float]:
    """Per-bin PSI terms, in bin order.

    Reported alongside the scalar because the scalar alone hides which bin drives it. On the locked
    reference, one bin routinely contributes the overwhelming majority, and an operator shown only
    a total would reasonably misread that as a broad shift.
    """

    if len(expected) != len(actual):
        raise ValueError("drift_bin_count_mismatch")

    terms: list[float] = []
    for expected_share, actual_share in zip(expected, actual, strict=True):
        floored_expected = max(expected_share, epsilon)
        floored_actual = max(actual_share, epsilon)
        terms.append(
            (floored_actual - floored_expected) * math.log(floored_actual / floored_expected)
        )
    return terms


def population_stability_index(
    expected: list[float], actual: list[float], epsilon: float = PSI_EPSILON
) -> float:
    """PSI between two share vectors already normalised to sum to one."""

    return sum(contributions(expected, actual, epsilon))


def from_counts(
    expected_counts: list[int], actual_counts: list[int], epsilon: float = PSI_EPSILON
) -> tuple[float, list[float]]:
    """Convenience wrapper: normalise both count vectors, then score."""

    expected = _shares(expected_counts)
    actual = _shares(actual_counts)
    return population_stability_index(expected, actual, epsilon), contributions(
        expected, actual, epsilon
    )
