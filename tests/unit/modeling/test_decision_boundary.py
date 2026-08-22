"""Regression fixture for the known Day 4 threshold tie-cluster condition.

This deliberately runs against the real locked Day 4 artifacts rather than a synthetic stand-in.
The condition is already known, understood and reproducible, so the actual model is the strongest
available fixture: if a future re-lock changes it, this test notices.

The artifacts live under the ignored ``artifacts/`` tree, so the test skips when they are absent
(a fresh clone) rather than failing. The synthetic tests below always run and cover the shape and
the non-fatal contract regardless.
"""

from pathlib import Path

import numpy as np
import pytest

from riskloom.modeling.artifacts import load_locked_model
from riskloom.modeling.config import ModelingConfig
from riskloom.modeling.data import PartitionData, TrainingData, load_training_data
from riskloom.modeling.model import LockedModel, LogisticPortableModel, PlattModel
from riskloom.modeling.training import decision_boundary_diagnostic

from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES  # isort: skip

SIMULATION = Path("artifacts/simulations/development-v1.1.0-config-bound-a")
FEATURES = Path("artifacts/features/development-v1.1.0-config-bound")
MODEL = Path("artifacts/models/development")

_ARTIFACTS_PRESENT = all(
    path.exists()
    for path in (
        SIMULATION / "labels.jsonl",
        FEATURES / "features.jsonl",
        MODEL / "model.json",
    )
)
requires_locked_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_PRESENT,
    reason="locked Day 4 artifacts are Git-ignored and absent in this checkout",
)


@pytest.fixture(scope="module")
def locked_diagnostic(modeling_config: ModelingConfig) -> dict[str, object]:
    model, _, _ = load_locked_model(MODEL, modeling_config)
    loaded = load_training_data(
        SIMULATION, FEATURES, modeling_config, include_held_out_sample=False
    )
    assert isinstance(loaded, TrainingData)
    return decision_boundary_diagnostic(model, loaded)


@requires_locked_artifacts
def test_locked_day_four_threshold_sits_on_a_tie_cluster_boundary(
    locked_diagnostic: dict[str, object],
) -> None:
    """The known condition, pinned exactly.

    The locked threshold is not one of the probabilities portable inference produces. It sits one
    unit in the last place above a cluster of eight rows, so those eight are allowed under portable
    inference even though Day 4's recorded confusion matrix counted them as denied.
    """

    assert locked_diagnostic["partition"] == "policy_selection"
    assert locked_diagnostic["threshold_is_an_observed_value"] is False
    assert locked_diagnostic["rows_tied_at_that_value"] == {
        "attack": 1,
        "legitimate": 7,
        "total": 8,
    }
    assert locked_diagnostic["distinct_probability_count"] == 15
    assert locked_diagnostic["decision_threshold"] == 0.0033862949155182734
    assert locked_diagnostic["nearest_observed_probability_below_threshold"] == (
        0.003386294915518273
    )


@requires_locked_artifacts
def test_locked_threshold_gap_is_far_inside_the_probability_parity_tolerance(
    locked_diagnostic: dict[str, object], modeling_config: ModelingConfig
) -> None:
    """Why probability parity cannot catch this: the gap is nine orders below the tolerance."""

    threshold = float(locked_diagnostic["decision_threshold"])  # type: ignore[arg-type]
    nearest = float(
        locked_diagnostic["nearest_observed_probability_below_threshold"]  # type: ignore[arg-type]
    )
    gap = threshold - nearest
    assert 0.0 < gap < modeling_config.parity_absolute_tolerance
    assert gap < 1e-15


def _partition(probabilities: np.ndarray, targets: np.ndarray) -> PartitionData:
    rows = probabilities.shape[0]
    return PartitionData(
        features=np.zeros((rows, FEATURE_COUNT), dtype=np.float64),
        targets=targets,
        scenarios=tuple("normal" for _ in range(rows)),
        campaign_ids=tuple(None for _ in range(rows)),
        occurred_at_ms=np.arange(rows, dtype=np.int64),
    )


def _training_data(probabilities: np.ndarray, targets: np.ndarray) -> TrainingData:
    partition = _partition(probabilities, targets)
    return TrainingData(
        train=partition,
        calibration_fit=partition,
        policy_selection=partition,
        boundary_timestamp="2026-01-24T00:00:00.000Z",
    )


def _model(threshold: float) -> LockedModel:
    return LockedModel(
        model_id="a" * 64,
        feature_order=list(FEATURE_NAMES),
        class_order=[0, 1],
        decision_threshold=threshold,
        candidate=LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=[0.0] * FEATURE_COUNT,
            intercept=0.0,
            scaler_mean=[0.0] * FEATURE_COUNT,
            scaler_scale=[1.0] * FEATURE_COUNT,
        ),
        calibration=PlattModel(coefficient=1.0, intercept=0.0, probability_clip_epsilon=1e-15),
    )


def test_diagnostic_reports_a_stable_threshold_when_it_is_an_observed_value() -> None:
    # All-zero features and coefficients give every row probability 0.5 exactly.
    targets = np.asarray([1, 0, 1, 0], dtype=np.int8)
    diagnostic = decision_boundary_diagnostic(_model(0.5), _training_data(np.zeros(4), targets))
    assert diagnostic["threshold_is_an_observed_value"] is True
    assert diagnostic["distinct_probability_count"] == 1
    assert diagnostic["nearest_observed_probability_below_threshold"] is None
    assert diagnostic["rows_tied_at_that_value"] == {"attack": 0, "legitimate": 0, "total": 0}


def test_diagnostic_splits_tied_rows_by_class() -> None:
    targets = np.asarray([1, 1, 0, 0, 0], dtype=np.int8)
    diagnostic = decision_boundary_diagnostic(_model(0.75), _training_data(np.zeros(5), targets))
    # Every row scores 0.5, which is the nearest observed value below a 0.75 threshold.
    assert diagnostic["threshold_is_an_observed_value"] is False
    assert diagnostic["nearest_observed_probability_below_threshold"] == 0.5
    assert diagnostic["rows_tied_at_that_value"] == {"attack": 2, "legitimate": 3, "total": 5}


def test_diagnostic_carries_a_plain_language_note_and_never_raises_a_verdict() -> None:
    diagnostic = decision_boundary_diagnostic(
        _model(0.5), _training_data(np.zeros(3), np.asarray([1, 0, 0], dtype=np.int8))
    )
    note = str(diagnostic["note"])
    assert "parity" in note
    assert "diagnostic, never a pass or fail condition" in note
    # No verdict-shaped key: this must not read like a gate.
    assert not {"status", "valid", "passed", "failed"}.intersection(diagnostic)
