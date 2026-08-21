import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES
from riskloom.modeling.artifacts import (
    ModelingArtifactError,
    load_locked_model,
    publish_model,
    runtime_dependency_versions,
)
from riskloom.modeling.canonical import canonical_json_bytes
from riskloom.modeling.config import ModelingConfig
from riskloom.modeling.data import TrainingData
from riskloom.modeling.model import (
    FittedCandidate,
    GradientBoostingPortableModel,
    LockedModel,
    LogisticPortableModel,
    ModelingFitError,
    PlattModel,
    PortableTree,
    balanced_sample_weights,
    export_candidate,
    fit_candidates,
    parity_result,
    portable_probabilities,
    portable_raw_scores,
)
from riskloom.modeling.training import TrainingOutputs

_OUT_OF_BOUNDS_CHILD_TREE: dict[str, list[float | int]] = {
    "node_count": 2,
    "children_left": [5, -1],
    "children_right": [1, -1],
    "features": [0, -2],
    "thresholds": [0.0, -2.0],
    "values": [0.0, 1.0],
}
_CYCLIC_TREE: dict[str, list[float | int]] = {
    "node_count": 3,
    "children_left": [1, 0, -1],
    "children_right": [2, 2, -1],
    "features": [0, 1, -2],
    "thresholds": [0.0, 0.0, -2.0],
    "values": [0.0, 1.0, 2.0],
}
_OUT_OF_RANGE_FEATURE_TREE: dict[str, list[float | int]] = {
    "node_count": 3,
    "children_left": [1, -1, -1],
    "children_right": [2, -1, -1],
    "features": [FEATURE_COUNT, -2, -2],
    "thresholds": [0.0, -2.0, -2.0],
    "values": [0.0, 1.0, 2.0],
}
_NEGATIVE_CHILD_TREE: dict[str, list[float | int]] = {
    "node_count": 3,
    "children_left": [-3, -1, -1],
    "children_right": [2, -1, -1],
    "features": [0, -2, -2],
    "thresholds": [0.0, -2.0, -2.0],
    "values": [0.0, 1.0, 2.0],
}
_SHARED_CHILD_TREE: dict[str, list[float | int]] = {
    "node_count": 6,
    "children_left": [1, 3, 3, -1, -1, -1],
    "children_right": [2, 4, 5, -1, -1, -1],
    "features": [0, 0, 0, -2, -2, -2],
    "thresholds": [0.0, 0.0, 0.0, -2.0, -2.0, -2.0],
    "values": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
}


def _gradient_locked_model() -> LockedModel:
    return LockedModel(
        model_id="b" * 64,
        feature_order=list(FEATURE_NAMES),
        class_order=[0, 1],
        decision_threshold=0.5,
        candidate=GradientBoostingPortableModel(
            candidate_name="gradient_boosting",
            learning_rate=0.5,
            trees=[
                PortableTree(
                    node_count=3,
                    children_left=[1, -1, -1],
                    children_right=[2, -1, -1],
                    features=[0, -2, -2],
                    thresholds=[0.5, -2.0, -2.0],
                    values=[0.0, -1.0, 2.0],
                )
            ],
        ),
        calibration=PlattModel(coefficient=1.0, intercept=0.0, probability_clip_epsilon=1e-15),
    )


def _logistic_locked_model() -> LockedModel:
    return LockedModel(
        model_id="a" * 64,
        feature_order=list(FEATURE_NAMES),
        class_order=[0, 1],
        decision_threshold=0.5,
        candidate=LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=[0.0] * FEATURE_COUNT,
            intercept=0.0,
            scaler_mean=[0.0] * FEATURE_COUNT,
            scaler_scale=[1.0] * FEATURE_COUNT,
        ),
        calibration=PlattModel(coefficient=1.0, intercept=0.0, probability_clip_epsilon=1e-15),
    )


def test_balanced_weights_use_train_counts_only() -> None:
    weights, values = balanced_sample_weights(np.asarray([0, 0, 0, 1], dtype=np.int8))
    assert values == {"legitimate": 2 / 3, "attack": 2.0}
    assert weights.tolist() == [2 / 3, 2 / 3, 2 / 3, 2.0]
    with pytest.raises(ModelingFitError, match="both_classes"):
        balanced_sample_weights(np.asarray([0, 0], dtype=np.int8))


def test_both_candidates_fit_calibrate_export_and_match_portable(
    tiny_training_data: TrainingData, modeling_config: ModelingConfig
) -> None:
    candidates, _ = fit_candidates(
        tiny_training_data.train.features,
        tiny_training_data.train.targets,
        tiny_training_data.calibration_fit.features,
        tiny_training_data.calibration_fit.targets,
        modeling_config,
    )
    assert [candidate.name for candidate in candidates] == [
        "logistic_regression",
        "gradient_boosting",
    ]
    assert all(
        np.isfinite(candidate.probabilities(tiny_training_data.policy_selection.features)).all()
        for candidate in candidates
    )


def test_locked_winner_has_exact_feature_order_and_parity(
    training_outputs: TrainingOutputs, tiny_training_data: TrainingData
) -> None:
    assert training_outputs.model.feature_order == list(FEATURE_NAMES)
    assert len(training_outputs.model.feature_order) == 75
    result = parity_result(
        training_outputs.winner,
        training_outputs.model,
        tiny_training_data.policy_selection.features,
        1e-10,
    )
    assert result["checked_row_count"] == tiny_training_data.policy_selection.row_count


def test_portable_logistic_inference_is_strict_and_stable() -> None:
    model = _logistic_locked_model()
    probabilities = portable_probabilities(model, np.zeros((2, FEATURE_COUNT)))
    assert probabilities.tolist() == [0.5, 0.5]
    with pytest.raises(ModelingFitError, match="feature_matrix"):
        portable_raw_scores(model.candidate, np.zeros((2, FEATURE_COUNT - 1)))
    with pytest.raises(ModelingFitError, match="feature_matrix"):
        portable_raw_scores(model.candidate, np.full((1, FEATURE_COUNT), np.nan))


def test_portable_tree_inference_and_topology_validation() -> None:
    tree = PortableTree(
        node_count=3,
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
        features=[0, -2, -2],
        thresholds=[0.5, -2.0, -2.0],
        values=[0.0, -1.0, 2.0],
    )
    candidate = GradientBoostingPortableModel(
        candidate_name="gradient_boosting", learning_rate=0.5, trees=[tree]
    )
    features = np.zeros((2, FEATURE_COUNT))
    features[1, 0] = 1.0
    assert portable_raw_scores(candidate, features).tolist() == [-0.5, 1.0]
    with pytest.raises(ValidationError):
        PortableTree(
            node_count=1,
            children_left=[0],
            children_right=[0],
            features=[0],
            thresholds=[0.0],
            values=[0.0],
        )


def test_portable_schema_rejects_dimensions_and_feature_order() -> None:
    with pytest.raises(ValidationError):
        LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=[0.0],
            intercept=0.0,
            scaler_mean=[0.0],
            scaler_scale=[1.0],
        )
    raw = _logistic_locked_model().model_dump(mode="python")
    raw["feature_order"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValidationError):
        LockedModel.model_validate(raw, strict=True)


@pytest.mark.parametrize(
    "tree",
    [
        {
            "node_count": 1,
            "children_left": [],
            "children_right": [],
            "features": [],
            "thresholds": [],
            "values": [],
        },
        {
            "node_count": 1,
            "children_left": [-1],
            "children_right": [-1],
            "features": [0],
            "thresholds": [0.0],
            "values": [0.0],
        },
        {
            "node_count": 2,
            "children_left": [1, -1],
            "children_right": [-1, -1],
            "features": [0, -2],
            "thresholds": [0.0, -2.0],
            "values": [0.0, 1.0],
        },
        {
            "node_count": 2,
            "children_left": [-1, -1],
            "children_right": [-1, -1],
            "features": [-2, -2],
            "thresholds": [-2.0, -2.0],
            "values": [0.0, 1.0],
        },
        pytest.param(_OUT_OF_BOUNDS_CHILD_TREE, id="out_of_bounds_child_index"),
        pytest.param(_CYCLIC_TREE, id="multi_node_cycle"),
        pytest.param(_OUT_OF_RANGE_FEATURE_TREE, id="split_feature_out_of_range"),
        pytest.param(_NEGATIVE_CHILD_TREE, id="negative_child_index_other_than_minus_one"),
        pytest.param(_SHARED_CHILD_TREE, id="reused_child_node_dag"),
    ],
)
def test_portable_tree_rejects_malformed_topologies(tree: dict[str, list[float | int]]) -> None:
    with pytest.raises(ValidationError):
        PortableTree.model_validate(tree, strict=True)


@pytest.mark.parametrize(
    "tree",
    [
        pytest.param(_OUT_OF_BOUNDS_CHILD_TREE, id="out_of_bounds_child_index"),
        pytest.param(_CYCLIC_TREE, id="multi_node_cycle"),
        pytest.param(_OUT_OF_RANGE_FEATURE_TREE, id="split_feature_out_of_range"),
        pytest.param(_NEGATIVE_CHILD_TREE, id="negative_child_index_other_than_minus_one"),
        pytest.param(_SHARED_CHILD_TREE, id="reused_child_node_dag"),
    ],
)
def test_locked_model_loader_rejects_malformed_tree_topologies(
    tmp_path: Path,
    modeling_config: ModelingConfig,
    training_outputs: TrainingOutputs,
    tree: dict[str, list[float | int]],
) -> None:
    output = tmp_path / "model"
    publish_model(
        output,
        _gradient_locked_model(),
        training_outputs.report,
        modeling_config,
        runtime_dependency_versions(),
    )
    tampered = json.loads((output / "model.json").read_bytes())
    tampered["candidate"]["trees"] = [tree]
    (output / "model.json").write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ModelingArtifactError, match="locked_model_schema_invalid"):
        load_locked_model(output, modeling_config)


def test_gradient_model_requires_tree_and_logistic_requires_positive_scale() -> None:
    with pytest.raises(ValidationError):
        GradientBoostingPortableModel(
            candidate_name="gradient_boosting", learning_rate=0.1, trees=[]
        )
    with pytest.raises(ValidationError):
        LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=[0.0] * FEATURE_COUNT,
            intercept=0.0,
            scaler_mean=[0.0] * FEATURE_COUNT,
            scaler_scale=[0.0] * FEATURE_COUNT,
        )


def test_export_rejects_candidate_type_mismatch(training_outputs: TrainingOutputs) -> None:
    invalid_logistic = FittedCandidate(
        name="logistic_regression",
        estimator=training_outputs.winner.estimator,
        scaler=None,
        calibrator=training_outputs.winner.calibrator,
    )
    with pytest.raises(ModelingFitError, match="logistic_candidate_state"):
        export_candidate(invalid_logistic)


def test_parity_failure_aborts_and_negative_logit_is_stable(
    training_outputs: TrainingOutputs, tiny_training_data: TrainingData
) -> None:
    raw = training_outputs.model.model_dump(mode="python")
    raw["calibration"]["intercept"] += 10.0
    changed = LockedModel.model_validate(raw, strict=True)
    with pytest.raises(ModelingFitError, match="parity_failed"):
        parity_result(
            training_outputs.winner,
            changed,
            tiny_training_data.policy_selection.features,
            1e-10,
        )
    model = _logistic_locked_model()
    raw = model.model_dump(mode="python")
    raw["calibration"]["intercept"] = -1_000.0
    negative = LockedModel.model_validate(raw, strict=True)
    assert portable_probabilities(negative, np.zeros((1, FEATURE_COUNT)))[0] == 1e-15
