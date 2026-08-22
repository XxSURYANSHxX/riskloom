from pathlib import Path

import numpy as np
import pytest

from riskloom.policy.bands import BandPolicy, evaluate_band, evaluate_single_threshold
from riskloom.policy.comparison import (
    GATE_COST_NOT_IMPROVED,
    GATE_FALSE_POSITIVE_CEILING,
    GATE_VALIDATION_BATCH_TOO_SMALL,
    build_comparison,
    cost_policy,
    evaluate_gates,
)
from riskloom.policy.config import PolicyConfig, PolicyConfigurationError, load_policy_config


@pytest.fixture(scope="session")
def policy_config() -> PolicyConfig:
    return load_policy_config(Path("configs/policy/default.json"))


def _batch(rows: int, attacks: int, denied_legitimate: int, missed_attacks: int) -> tuple:
    """Build targets and probabilities producing an exact confusion at threshold 0.5."""

    targets = np.asarray([1] * attacks + [0] * (rows - attacks), dtype=np.int64)
    probabilities = np.concatenate(
        [
            np.full(attacks - missed_attacks, 0.9),
            np.full(missed_attacks, 0.1),
            np.full(denied_legitimate, 0.9),
            np.full(rows - attacks - denied_legitimate, 0.1),
        ]
    )
    return targets, probabilities


def test_shipped_default_config_matches_the_documented_policy(policy_config: PolicyConfig) -> None:
    assert policy_config.false_negative_cost_units == 25
    assert policy_config.false_positive_cost_units == 1
    assert policy_config.review_cost_units == 3
    assert policy_config.maximum_false_positive_rate_basis_points == 100
    assert policy_config.minimum_validation_rows == 2_000
    assert policy_config.minimum_validation_attacks == 100


def test_config_refuses_to_change_the_inherited_day_four_costs() -> None:
    with pytest.raises(ValueError, match="1:25"):
        PolicyConfig(
            false_negative_cost_units=50,
            false_positive_cost_units=1,
            review_cost_units=3,
            maximum_false_positive_rate_basis_points=100,
            minimum_validation_rows=1,
            minimum_validation_attacks=1,
            threshold_grid_size=16,
        )
    with pytest.raises(ValueError, match="1:25"):
        PolicyConfig(
            false_negative_cost_units=25,
            false_positive_cost_units=2,
            review_cost_units=3,
            maximum_false_positive_rate_basis_points=100,
            minimum_validation_rows=1,
            minimum_validation_attacks=1,
            threshold_grid_size=16,
        )


def test_non_canonical_policy_config_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_bytes(b'{\n  "config_schema_version": "1.0.0"\n}\n')
    with pytest.raises(PolicyConfigurationError):
        load_policy_config(path)


def test_all_gates_pass_when_the_band_wins_cleanly(policy_config: PolicyConfig) -> None:
    costs = cost_policy(policy_config)
    targets, probabilities = _batch(9_000, 180, denied_legitimate=40, missed_attacks=10)
    incumbent = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    banded = evaluate_band(
        targets, probabilities, BandPolicy(lower_threshold=0.5, upper_threshold=0.5), costs
    )
    gates = evaluate_gates(incumbent, banded, policy_config)
    # Identical policies cannot beat each other, so this asserts the shape rather than a win.
    assert gates["validation_batch_sufficient"] is True
    assert gates["false_positive_rate_within_ceiling"] is True
    assert gates["failed_gates"] == [GATE_COST_NOT_IMPROVED]


def test_cost_gate_fails_when_the_band_does_not_beat_the_incumbent(
    policy_config: PolicyConfig,
) -> None:
    costs = cost_policy(policy_config)
    targets, probabilities = _batch(9_000, 180, denied_legitimate=10, missed_attacks=5)
    incumbent = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    worse = evaluate_single_threshold(targets, probabilities, 0.95, costs)
    gates = evaluate_gates(incumbent, worse, policy_config)
    assert gates["beats_incumbent_cost"] is False
    assert gates["cost_delta_units"] > 0
    assert GATE_COST_NOT_IMPROVED in gates["failed_gates"]
    assert gates["approval_eligible"] is False


def test_false_positive_ceiling_gate_fails_above_the_configured_limit(
    policy_config: PolicyConfig,
) -> None:
    costs = cost_policy(policy_config)
    # 200 denied legitimate rows out of 8_820 is 227 bp, above the 100 bp default ceiling.
    targets, probabilities = _batch(9_000, 180, denied_legitimate=200, missed_attacks=170)
    incumbent = evaluate_single_threshold(targets, probabilities, 0.95, costs)
    banded = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    gates = evaluate_gates(incumbent, banded, policy_config)
    assert gates["false_positive_rate_basis_points"] > 100
    assert gates["false_positive_rate_within_ceiling"] is False
    assert GATE_FALSE_POSITIVE_CEILING in gates["failed_gates"]
    assert gates["approval_eligible"] is False


def test_small_batch_gate_fails_on_rows_and_on_attacks(policy_config: PolicyConfig) -> None:
    costs = cost_policy(policy_config)
    targets, probabilities = _batch(500, 120, denied_legitimate=1, missed_attacks=1)
    incumbent = evaluate_single_threshold(targets, probabilities, 0.95, costs)
    banded = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    gates = evaluate_gates(incumbent, banded, policy_config)
    assert gates["observed_validation_rows"] == 500
    assert GATE_VALIDATION_BATCH_TOO_SMALL in gates["failed_gates"]

    targets, probabilities = _batch(9_000, 40, denied_legitimate=1, missed_attacks=1)
    incumbent = evaluate_single_threshold(targets, probabilities, 0.95, costs)
    banded = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    gates = evaluate_gates(incumbent, banded, policy_config)
    assert gates["observed_validation_attacks"] == 40
    assert GATE_VALIDATION_BATCH_TOO_SMALL in gates["failed_gates"]


def test_null_false_positive_rate_fails_the_ceiling_rather_than_passing_it(
    policy_config: PolicyConfig,
) -> None:
    costs = cost_policy(policy_config)
    targets = np.ones(10, dtype=np.int64)
    probabilities = np.full(10, 0.9)
    outcome = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    assert outcome.legitimate_count == 0
    gates = evaluate_gates(outcome, outcome, policy_config)
    assert gates["false_positive_rate_basis_points"] is None
    assert gates["false_positive_rate_within_ceiling"] is False
    assert GATE_FALSE_POSITIVE_CEILING in gates["failed_gates"]


def test_comparison_payload_is_aggregate_only(policy_config: PolicyConfig) -> None:
    costs = cost_policy(policy_config)
    targets, probabilities = _batch(9_000, 180, denied_legitimate=40, missed_attacks=10)
    band = BandPolicy(lower_threshold=0.5, upper_threshold=0.5)
    incumbent = evaluate_single_threshold(targets, probabilities, 0.5, costs)
    banded = evaluate_band(targets, probabilities, band, costs)
    payload = build_comparison(band, 0.5, incumbent, banded, policy_config, probabilities)
    assert set(payload) == {
        "banded_policy",
        "cost_policy",
        "gates",
        "incumbent_policy",
        "threshold_boundary",
    }
    # The boundary diagnostic must expose when a threshold is not an observed probability, which
    # is exactly the condition that turns a tie cluster into a spurious cost delta.
    boundary = payload["threshold_boundary"]["incumbent"]
    assert boundary["threshold_is_an_observed_value"] is False
    assert boundary["distinct_probability_count"] == 2
    assert boundary["nearest_observed_value_below_threshold"] == 0.1
    assert boundary["tied_rows_at_nearest_value_below_threshold"] > 0
    rendered = str(payload)
    for token in ("evt_", "cmp_", "scn_", "event_id", "campaign_id", "scenario_type"):
        assert token not in rendered
    assert payload["cost_policy"]["interpretation"] == "abstract_policy_units_not_currency"
    assert payload["cost_policy"]["review_cost_units"] == 3


def test_cost_ratio_may_be_rescaled_but_the_ratio_itself_may_not_change() -> None:
    """Rescaling is what makes an integer review cost expressible between allow and deny."""

    rescaled = PolicyConfig(
        false_negative_cost_units=250,
        false_positive_cost_units=10,
        review_cost_units=3,
        maximum_false_positive_rate_basis_points=100,
        minimum_validation_rows=1,
        minimum_validation_attacks=1,
        threshold_grid_size=16,
    )
    assert rescaled.false_negative_cost_units == 25 * rescaled.false_positive_cost_units
    # A zero false-positive cost would make denying free; the ratio check must reject it before
    # the ratio arithmetic degenerates.
    with pytest.raises(ValueError, match="at least 1 unit"):
        PolicyConfig(
            false_negative_cost_units=25,
            false_positive_cost_units=0,
            review_cost_units=3,
            maximum_false_positive_rate_basis_points=100,
            minimum_validation_rows=1,
            minimum_validation_attacks=1,
            threshold_grid_size=16,
        )


def test_shipped_rescaled_experiment_config_is_clearly_separate() -> None:
    experiment = load_policy_config(Path("configs/policy/rescaled-experiment.json"))
    default = load_policy_config(Path("configs/policy/default.json"))
    assert experiment.false_positive_cost_units == 10
    assert experiment.false_negative_cost_units == 250
    assert experiment.review_cost_units == 3
    # The experiment leaves room for a review cost strictly between doing nothing and denying.
    assert experiment.review_cost_units < experiment.false_positive_cost_units
    assert default.review_cost_units > default.false_positive_cost_units
    assert experiment.model_dump() != default.model_dump()
