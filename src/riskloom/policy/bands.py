"""Two-threshold ALLOW / REVIEW / DENY banding and its deterministic cost-minimising sweep.

Policy inputs are exactly the locked model's calibrated probability per event. Ground-truth targets
appear in this module only as an evaluation array supplied by an offline scoring caller so that a
cost can be computed; they are never consulted to route an event. ``band_decisions`` -- the
function that would run at inference time -- takes probabilities and a band, and nothing else.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

ALLOW = "allow"
REVIEW = "review"
DENY = "deny"

ALLOW_CODE = 0
REVIEW_CODE = 1
DENY_CODE = 2

DECISION_NAMES = (ALLOW, REVIEW, DENY)

# The extended deterministic tie-break ladder for the two-threshold case, applied in this order to
# every band that ties on the level above it. Day 4's single-threshold ladder was
# (cost, false positive, -true positive, -precision, -threshold). Precision is deliberately absent
# here: precision is exactly true_positive / (true_positive + false_positive), so any two bands
# that have already tied on both false positive and true positive necessarily tie on precision too.
# That rung can never change an outcome, as the Gate B1 mutation analysis established, so extending
# it into the band case would add a step unreachable by construction. Review count takes its place,
# and the two threshold values plus the stable candidate-index pair make the ordering total.
TIE_BREAK_ORDER = (
    "minimum_cost",
    "minimum_false_positive",
    "maximum_true_positive",
    "minimum_review_count",
    "maximum_upper_threshold",
    "maximum_lower_threshold",
    "stable_candidate_index",
)


class PolicyBandError(ValueError):
    """A safe banding or sweep error."""


class CostPolicy(BaseModel):
    """Abstract cost weights. See :class:`riskloom.policy.config.PolicyConfig` for framing."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    false_negative_cost_units: int = Field(ge=0)
    false_positive_cost_units: int = Field(ge=0)
    review_cost_units: int = Field(ge=0)


class BandPolicy(BaseModel):
    """Locked two-threshold band.

    ``probability < lower_threshold`` allows, ``lower_threshold <= probability < upper_threshold``
    reviews, and ``probability >= upper_threshold`` denies. A band whose thresholds are equal has an
    empty review tier and is exactly the Day 4 single-threshold policy.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, strict=True, allow_inf_nan=False
    )

    lower_threshold: float = Field(ge=0.0, le=1.0)
    upper_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> Self:
        if self.lower_threshold > self.upper_threshold:
            raise ValueError("policy band lower threshold must not exceed upper threshold")
        return self

    @property
    def has_review_tier(self) -> bool:
        return self.upper_threshold > self.lower_threshold


@dataclass(frozen=True, slots=True)
class BandOutcome:
    """Confusion counts and cost for one band over one scored batch."""

    allow_count: int
    review_count: int
    deny_count: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    row_count: int
    attack_count: int
    legitimate_count: int
    cost_units: int

    def as_report(self) -> dict[str, float | int | None]:
        legitimate = self.legitimate_count
        return {
            "allow_count": self.allow_count,
            "attack_count": self.attack_count,
            "cost_units": self.cost_units,
            "deny_count": self.deny_count,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "false_positive_rate": _rate(self.false_positive, legitimate),
            "false_positive_rate_basis_points": _basis_points(self.false_positive, legitimate),
            "false_positives_per_10000_legitimate": _rate(self.false_positive * 10_000, legitimate),
            "legitimate_count": legitimate,
            "review_count": self.review_count,
            "review_rate": _rate(self.review_count, self.row_count),
            "row_count": self.row_count,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
        }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _basis_points(numerator: int, denominator: int) -> int | None:
    """Integer basis points, rounded half-up, so ceiling comparisons never use floats."""

    if not denominator:
        return None
    return (numerator * 20_000 + denominator) // (2 * denominator)


def boundary_diagnostics(probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    """Describe how fragile a threshold is against the scored probability distribution.

    A tree ensemble emits very few distinct raw scores, so calibrated probabilities arrive in large
    ties. When a threshold is not itself one of those observed values, the rows in the nearest tie
    cluster below it all fall on the allow side, and a difference far smaller than any parity
    tolerance decides the fate of every row in that cluster at once. Reporting a cost delta without
    this context can present a floating-point artifact as a policy improvement, so the comparison
    carries these fields explicitly.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise PolicyBandError("policy_probability_vector_invalid")
    observed = np.unique(values)
    below = observed[observed < threshold]
    nearest = float(below.max()) if below.size else None
    tied = int(np.count_nonzero(values == nearest)) if nearest is not None else 0
    return {
        "distinct_probability_count": int(observed.size),
        "nearest_observed_value_below_threshold": nearest,
        "threshold_is_an_observed_value": bool(np.any(values == threshold)),
        "tied_rows_at_nearest_value_below_threshold": tied,
    }


def band_decisions(probabilities: np.ndarray, band: BandPolicy) -> np.ndarray:
    """Route each calibrated probability to ALLOW (0), REVIEW (1) or DENY (2).

    This is the entire inference-time policy surface. It sees no label, no scenario type, no
    campaign identifier and no entity token.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise PolicyBandError("policy_probability_vector_invalid")
    decisions = np.full(values.shape[0], ALLOW_CODE, dtype=np.int8)
    decisions[values >= band.lower_threshold] = REVIEW_CODE
    decisions[values >= band.upper_threshold] = DENY_CODE
    return decisions


def _validate_scoring_inputs(targets: np.ndarray, probabilities: np.ndarray) -> None:
    if targets.shape != probabilities.shape or targets.ndim != 1:
        raise PolicyBandError("policy_scoring_shape_invalid")
    if not np.isfinite(probabilities).all():
        raise PolicyBandError("policy_probability_vector_invalid")
    if not bool(np.isin(targets, (0, 1)).all()):
        raise PolicyBandError("policy_target_vector_invalid")


def evaluate_band(
    targets: np.ndarray, probabilities: np.ndarray, band: BandPolicy, costs: CostPolicy
) -> BandOutcome:
    """Score one band. A reviewed event is neither a false positive nor a false negative.

    An attack routed to REVIEW was not allowed through, so it is not a false negative. A legitimate
    event routed to REVIEW was not denied, so it is not a false positive. Both still incur the
    review cost, which is why the review count is its own term in the total rather than folded into
    the false-positive term.
    """

    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    _validate_scoring_inputs(target_array, probability_array)
    decisions = band_decisions(probability_array, band)
    positive_target = target_array == 1
    allowed = decisions == ALLOW_CODE
    reviewed = decisions == REVIEW_CODE
    denied = decisions == DENY_CODE
    true_positive = int(np.count_nonzero(denied & positive_target))
    false_positive = int(np.count_nonzero(denied & ~positive_target))
    false_negative = int(np.count_nonzero(allowed & positive_target))
    true_negative = int(np.count_nonzero(allowed & ~positive_target))
    review_count = int(np.count_nonzero(reviewed))
    return BandOutcome(
        allow_count=int(np.count_nonzero(allowed)),
        review_count=review_count,
        deny_count=int(np.count_nonzero(denied)),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        row_count=int(target_array.shape[0]),
        attack_count=int(np.count_nonzero(positive_target)),
        legitimate_count=int(np.count_nonzero(~positive_target)),
        cost_units=(
            false_negative * costs.false_negative_cost_units
            + false_positive * costs.false_positive_cost_units
            + review_count * costs.review_cost_units
        ),
    )


def evaluate_single_threshold(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float, costs: CostPolicy
) -> BandOutcome:
    """Score the Day 4 policy as the degenerate band with an empty review tier."""

    band = BandPolicy(lower_threshold=threshold, upper_threshold=threshold)
    return evaluate_band(targets, probabilities, band, costs)


def sweep_candidates(
    probabilities: np.ndarray, grid_size: int, required: tuple[float, ...] = ()
) -> tuple[float, ...]:
    """Deterministic threshold candidates.

    Day 4 swept the observed probability values rather than an even grid over ``[0, 1]``, because
    the calibrated distribution is extremely skewed toward zero and a uniform grid would be coarser
    than the entire region where the decision actually lives. That choice is preserved here. When
    the observed set is larger than ``grid_size`` it is subsampled by evenly spaced index using the
    same integer formula Day 4 used, which is deterministic and independent of data order. Values in
    ``required`` are always retained, so the sweep can always express the incumbent policy exactly.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise PolicyBandError("policy_probability_vector_invalid")
    if grid_size < 2:
        raise PolicyBandError("policy_grid_size_invalid")
    observed = sorted({float(value) for value in values} | {0.0, 1.0})
    total = len(observed)
    if total > grid_size:
        indexes = sorted({(index * (total - 1)) // (grid_size - 1) for index in range(grid_size)})
        observed = [observed[index] for index in indexes]
    return tuple(sorted(set(observed) | {float(value) for value in required}))


_SweepEntry = tuple[int, int, int, int, float, float, int, int]


def _tie_break_key(entry: _SweepEntry) -> tuple[int, int, int, int, float, float, int, int]:
    cost, false_positive, true_positive, review_count, lower, upper, lower_index, upper_index = (
        entry
    )
    return (
        cost,
        false_positive,
        -true_positive,
        review_count,
        -upper,
        -lower,
        lower_index,
        upper_index,
    )


def select_band(
    targets: np.ndarray,
    probabilities: np.ndarray,
    costs: CostPolicy,
    grid_size: int,
    required: tuple[float, ...] = (),
) -> tuple[BandPolicy, BandOutcome]:
    """Fit both thresholds by exhaustive deterministic sweep over the candidate grid.

    Fitting happens on the ``policy_selection`` partition. That partition's designated Day 4 role
    was already candidate and threshold selection -- it is the slice set aside precisely so a
    decision rule could be chosen without touching held-out data. Choosing two thresholds on it
    instead of one is an extension of that same designated role, not new leakage: no held-out row
    and no counterfactual-validation row is read here.
    """

    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    _validate_scoring_inputs(target_array, probability_array)
    if target_array.shape[0] == 0:
        raise PolicyBandError("policy_band_requires_rows")

    candidates = sweep_candidates(probability_array, grid_size, required)
    order = np.argsort(probability_array, kind="stable")
    sorted_probabilities = probability_array[order]
    sorted_attacks = (target_array[order] == 1).astype(np.int64)
    cumulative_attacks = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sorted_attacks)))
    row_count = int(sorted_probabilities.shape[0])
    total_attacks = int(cumulative_attacks[-1])

    boundaries = np.searchsorted(sorted_probabilities, np.asarray(candidates), side="left")
    lower_index = boundaries[:, None]
    upper_index = boundaries[None, :]
    review_count = upper_index - lower_index
    deny_count = row_count - upper_index
    attacks_allowed = cumulative_attacks[lower_index]
    attacks_denied = total_attacks - cumulative_attacks[upper_index]
    # Denials, and therefore false positives and true positives, depend only on the upper
    # threshold; review count is the only quantity that depends on both. Keeping these as
    # (1, m) rows rather than broadcasting to (m, m) is what makes the full pair sweep cheap.
    false_positive = deny_count - attacks_denied
    cost = (
        attacks_allowed * costs.false_negative_cost_units
        + false_positive * costs.false_positive_cost_units
        + review_count * costs.review_cost_units
    )
    masked = np.where(lower_index <= upper_index, cost, np.iinfo(np.int64).max)
    minimum_cost = int(masked.min())

    entries: list[_SweepEntry] = []
    for pair in np.argwhere(masked == minimum_cost):
        low = int(pair[0])
        high = int(pair[1])
        entries.append(
            (
                minimum_cost,
                int(false_positive[0, high]),
                int(attacks_denied[0, high]),
                int(review_count[low, high]),
                candidates[low],
                candidates[high],
                low,
                high,
            )
        )
    best = min(entries, key=_tie_break_key)
    band = BandPolicy(lower_threshold=best[4], upper_threshold=best[5])
    outcome = evaluate_band(target_array, probability_array, band, costs)
    if outcome.cost_units != minimum_cost:
        raise PolicyBandError("policy_band_sweep_inconsistent")
    return band, outcome
