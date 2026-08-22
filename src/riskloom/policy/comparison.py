"""Counterfactual comparison of the banded policy against the incumbent single threshold.

Both policies are scored on exactly the same rows with exactly the same cost function, so the
comparison is a like-for-like counterfactual rather than two differently-measured numbers. The
batch this runs on is only ever read: nothing here feeds back into band fitting.
"""

from typing import Any

import numpy as np

from riskloom.policy.bands import (
    BandOutcome,
    BandPolicy,
    CostPolicy,
    boundary_diagnostics,
)
from riskloom.policy.config import PolicyConfig

# Reasons a banded policy is not eligible for approval. Any one of these is disqualifying, and the
# CLI must refuse to mark the policy approved while any is present -- including when a human
# explicitly passes the approval flag.
GATE_COST_NOT_IMPROVED = "banded_policy_does_not_beat_incumbent_cost"
GATE_FALSE_POSITIVE_CEILING = "banded_policy_exceeds_false_positive_rate_ceiling"
GATE_VALIDATION_BATCH_TOO_SMALL = "validation_batch_below_minimum_evidence"


def cost_policy(config: PolicyConfig) -> CostPolicy:
    return CostPolicy(
        false_negative_cost_units=config.false_negative_cost_units,
        false_positive_cost_units=config.false_positive_cost_units,
        review_cost_units=config.review_cost_units,
    )


def evaluate_gates(
    incumbent: BandOutcome, banded: BandOutcome, config: PolicyConfig
) -> dict[str, Any]:
    """Decide whether the banded policy has earned the right to be approved by a human.

    This never activates anything. It reports whether the evidence clears every gate; a separate
    explicit human step is still required, and that step is refused when any gate fails.
    """

    ceiling = config.maximum_false_positive_rate_basis_points
    observed = banded.as_report()["false_positive_rate_basis_points"]
    # A batch with no legitimate rows yields a null rate; treat that as failing the ceiling gate
    # rather than silently passing it.
    within_ceiling = isinstance(observed, int) and observed <= ceiling
    sufficient = (
        banded.row_count >= config.minimum_validation_rows
        and banded.attack_count >= config.minimum_validation_attacks
    )
    beats_incumbent = banded.cost_units < incumbent.cost_units

    failures: list[str] = []
    if not beats_incumbent:
        failures.append(GATE_COST_NOT_IMPROVED)
    if not within_ceiling:
        failures.append(GATE_FALSE_POSITIVE_CEILING)
    if not sufficient:
        failures.append(GATE_VALIDATION_BATCH_TOO_SMALL)

    return {
        "approval_eligible": not failures,
        "beats_incumbent_cost": beats_incumbent,
        "cost_delta_units": banded.cost_units - incumbent.cost_units,
        "failed_gates": sorted(failures),
        "false_positive_rate_basis_points": observed,
        "false_positive_rate_ceiling_basis_points": ceiling,
        "false_positive_rate_within_ceiling": within_ceiling,
        "minimum_validation_attacks": config.minimum_validation_attacks,
        "minimum_validation_rows": config.minimum_validation_rows,
        "observed_validation_attacks": banded.attack_count,
        "observed_validation_rows": banded.row_count,
        "validation_batch_sufficient": sufficient,
    }


def build_comparison(
    band: BandPolicy,
    incumbent_threshold: float,
    incumbent: BandOutcome,
    banded: BandOutcome,
    config: PolicyConfig,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Aggregate-only comparison payload. No identifier or per-event value ever appears here."""

    gates = evaluate_gates(incumbent, banded, config)
    return {
        "threshold_boundary": {
            "banded_lower": boundary_diagnostics(probabilities, band.lower_threshold),
            "banded_upper": boundary_diagnostics(probabilities, band.upper_threshold),
            "incumbent": boundary_diagnostics(probabilities, incumbent_threshold),
            "interpretation": (
                "A cost delta driven entirely by a tie cluster between two thresholds is a"
                " floating-point boundary artifact, not a policy improvement."
            ),
        },
        "banded_policy": {
            "band": {
                "lower_threshold": band.lower_threshold,
                "upper_threshold": band.upper_threshold,
            },
            "has_review_tier": band.has_review_tier,
            "outcome": banded.as_report(),
        },
        "cost_policy": {
            "false_negative_cost_units": config.false_negative_cost_units,
            "false_positive_cost_units": config.false_positive_cost_units,
            "interpretation": "abstract_policy_units_not_currency",
            "review_cost_units": config.review_cost_units,
            "total_cost_formula": (
                "false_negative * false_negative_cost_units"
                " + false_positive * false_positive_cost_units"
                " + review_count * review_cost_units"
            ),
        },
        "gates": gates,
        "incumbent_policy": {
            "decision_threshold": incumbent_threshold,
            "outcome": incumbent.as_report(),
        },
    }
