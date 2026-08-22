from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from riskloom.features.schema import FEATURE_NAMES
from riskloom.modeling.artifacts import (
    PublicationResult,
    load_locked_model,
    model_identity_material,
    publish_model,
    require_output_separate,
    runtime_dependency_versions,
)
from riskloom.modeling.canonical import canonical_json_bytes, canonical_sha256
from riskloom.modeling.config import ModelingConfig, load_modeling_config
from riskloom.modeling.data import TrainingData, ValidationData, load_training_data
from riskloom.modeling.metrics import (
    complete_metrics,
    probability_metrics,
    reliability_metrics,
    select_threshold,
)
from riskloom.modeling.model import (
    FittedCandidate,
    LockedModel,
    export_candidate,
    export_platt,
    fit_candidates,
    parity_result,
    portable_probabilities,
)
from riskloom.policy.bands import boundary_diagnostics


class ModelingTrainingError(ValueError):
    """A safe training or validation error."""


@dataclass(frozen=True, slots=True)
class TrainingOutputs:
    model: LockedModel
    report: dict[str, Any]
    winner: FittedCandidate


def _sample(features: np.ndarray, maximum: int) -> np.ndarray:
    count = min(features.shape[0], maximum)
    if count == features.shape[0]:
        return features
    indexes = np.asarray(
        [(index * (features.shape[0] - 1)) // (count - 1) for index in range(count)],
        dtype=np.int64,
    )
    return features[indexes]


def _candidate_parameters(config: ModelingConfig) -> dict[str, Any]:
    return {
        "gradient_boosting": {
            **config.gradient_boosting.model_dump(mode="json"),
            "ccp_alpha": 0.0,
            "criterion_semantics": "friedman_mse_sklearn_default",
            "init": "zero",
            "minimum_weight_fraction_leaf": 0.0,
            "n_iter_no_change": None,
            "validation_fraction": 0.1,
            "verbose": 0,
            "warm_start": False,
        },
        "logistic_regression": {
            **config.logistic_regression.model_dump(mode="json"),
            "class_weight": None,
            "n_jobs": None,
            "penalty_semantics": "l2_via_l1_ratio_zero",
            "verbose": 0,
            "warm_start": False,
        },
        "platt_calibration": {
            **config.platt_calibration.model_dump(mode="json"),
            "class_weight": None,
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1.0,
            "l1_ratio": 0.0,
            "n_jobs": None,
            "penalty_semantics": "l2_via_l1_ratio_zero",
            "verbose": 0,
            "warm_start": False,
        },
    }


def _ranking_key(
    item: tuple[str, float, dict[str, Any]],
) -> tuple[float, int, int, float, float, str]:
    name, threshold, policy = item
    precision = policy["precision"]
    return (
        float(policy["cost_units"]),
        int(policy["false_positive"]),
        -int(policy["true_positive"]),
        -float(precision) if precision is not None else 0.0,
        -threshold,
        name,
    )


def _feature_importance(candidate: FittedCandidate) -> list[dict[str, float | str]]:
    estimator: Any = candidate.estimator
    if candidate.name == "logistic_regression":
        values = np.abs(estimator.coef_[0])
    else:
        values = estimator.feature_importances_
    ranked = sorted(
        zip(FEATURE_NAMES, (float(value) for value in values), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return [{"feature": name, "importance": value} for name, value in ranked]


def build_training_outputs(data: TrainingData, config: ModelingConfig) -> TrainingOutputs:
    fitted, class_weights = fit_candidates(
        data.train.features,
        data.train.targets,
        data.calibration_fit.features,
        data.calibration_fit.targets,
        config,
    )
    candidate_reports: dict[str, Any] = {}
    ranking: list[tuple[str, float, dict[str, Any]]] = []
    for candidate in fitted:
        calibration_probabilities = candidate.probabilities(data.calibration_fit.features)
        policy_probabilities = candidate.probabilities(data.policy_selection.features)
        threshold, policy = select_threshold(
            data.policy_selection.targets,
            policy_probabilities,
            config.false_positive_cost_units,
            config.false_negative_cost_units,
        )
        ranking.append((candidate.name, threshold, policy))
        candidate_reports[candidate.name] = {
            "calibration_fit_probability_metrics": probability_metrics(
                data.calibration_fit.targets, calibration_probabilities
            ),
            "calibration_fit_reliability": reliability_metrics(
                data.calibration_fit.targets,
                calibration_probabilities,
                config.reliability_bin_count,
            ),
            "feature_importance": {
                "interpretation": "model_importance_not_causal_explanation",
                "ranking": _feature_importance(candidate),
                "type": (
                    "absolute_standardized_coefficient"
                    if candidate.name == "logistic_regression"
                    else "impurity_based"
                ),
            },
            "policy_selection_metrics": complete_metrics(
                data.policy_selection.targets,
                policy_probabilities,
                threshold,
                config.false_positive_cost_units,
                config.false_negative_cost_units,
                config.reliability_bin_count,
                data.policy_selection.scenarios,
                data.policy_selection.campaign_ids,
                data.policy_selection.occurred_at_ms,
            ),
        }
    selected_name, selected_threshold, _ = min(ranking, key=_ranking_key)
    winner = next(candidate for candidate in fitted if candidate.name == selected_name)
    portable_candidate = export_candidate(winner)
    platt = export_platt(winner, config.probability_clip_epsilon)
    without_id = {
        "artifact_type": "locked_binary_risk_model",
        "calibration": platt.model_dump(mode="json"),
        "candidate": portable_candidate.model_dump(mode="json"),
        "class_order": [0, 1],
        "decision_threshold": selected_threshold,
        "feature_order": list(FEATURE_NAMES),
        "model_schema_version": "1.0.0",
        "product": "RiskLoom",
    }
    model_id = canonical_sha256(model_identity_material(without_id, config))
    model = LockedModel(
        model_id=model_id,
        calibration=platt,
        candidate=portable_candidate,
        class_order=[0, 1],
        decision_threshold=selected_threshold,
        feature_order=list(FEATURE_NAMES),
    )
    parity = {
        "calibration_fit": parity_result(
            winner,
            model,
            _sample(data.calibration_fit.features, config.parity_sample_rows),
            config.parity_absolute_tolerance,
        ),
        "policy_selection": parity_result(
            winner,
            model,
            _sample(data.policy_selection.features, config.parity_sample_rows),
            config.parity_absolute_tolerance,
        ),
        "train": parity_result(
            winner,
            model,
            _sample(data.train.features, config.parity_sample_rows),
            config.parity_absolute_tolerance,
        ),
    }
    report = {
        "artifact_type": "model_training_report",
        "boundary_timestamp": data.boundary_timestamp,
        "candidate_parameters": _candidate_parameters(config),
        "candidates": {name: candidate_reports[name] for name in sorted(candidate_reports)},
        "class_weights": class_weights,
        "feature_importance": {
            "interpretation": "model_importance_not_causal_explanation",
            "ranking": _feature_importance(winner),
            "type": (
                "absolute_standardized_coefficient"
                if winner.name == "logistic_regression"
                else "impurity_based"
            ),
        },
        "model_id": model.model_id,
        "parity": parity,
        "partitions": {
            "calibration_fit": {
                "attack_count": int(np.count_nonzero(data.calibration_fit.targets == 1)),
                "legitimate_count": int(np.count_nonzero(data.calibration_fit.targets == 0)),
                "row_count": data.calibration_fit.row_count,
            },
            "policy_selection": {
                "attack_count": int(np.count_nonzero(data.policy_selection.targets == 1)),
                "legitimate_count": int(np.count_nonzero(data.policy_selection.targets == 0)),
                "row_count": data.policy_selection.row_count,
            },
            "train": {
                "attack_count": int(np.count_nonzero(data.train.targets == 1)),
                "legitimate_count": int(np.count_nonzero(data.train.targets == 0)),
                "row_count": data.train.row_count,
            },
        },
        "product": "RiskLoom",
        "selected_candidate": selected_name,
        "selected_threshold": selected_threshold,
        "selection_policy": {
            "false_negative_cost_units": config.false_negative_cost_units,
            "false_positive_cost_units": config.false_positive_cost_units,
            "tie_break_order": [
                "minimum_cost",
                "minimum_false_positive",
                "maximum_true_positive",
                "maximum_precision",
                "maximum_threshold",
                "candidate_name",
            ],
        },
        "source": {
            "feature_dataset_id": config.source_contract.feature_dataset_id,
            "feature_engine_version": config.source_contract.feature_engine_version,
            "feature_schema_version": config.source_contract.feature_schema_version,
            "features_sha256": config.source_contract.features_sha256,
            "labels_sha256": config.source_contract.labels_sha256,
            "simulation_dataset_id": config.source_contract.simulation_dataset_id,
            "simulation_generator_version": (config.source_contract.simulation_generator_version),
        },
        "training_report_schema_version": "1.0.0",
    }
    return TrainingOutputs(model=model, report=report, winner=winner)


DECISION_BOUNDARY_NOTE = (
    "The locked decision threshold is compared against the calibrated probabilities that portable"
    " JSON inference actually produces. When the threshold is not itself one of those observed"
    " values, every row in the nearest tie cluster below it sits on the allow side, and any"
    " implementation whose rounding differs by less than the probability-parity tolerance can"
    " still disagree about the discrete decision for all of those rows at once. Probability parity"
    " passing therefore does not imply decision parity. This is reported for visibility on future"
    " re-locks and retraining runs; it is a diagnostic, never a pass or fail condition."
)


def decision_boundary_diagnostic(model: LockedModel, data: TrainingData) -> dict[str, Any]:
    """Report how fragile the locked threshold is against portable inference.

    Reported for ``policy_selection`` because that is the partition the threshold was selected on,
    so it is where a tie-cluster landing has any bearing on the recorded confusion matrix. This is
    purely additive: it computes no verdict, raises nothing of its own, and cannot change whether
    ``validate-model`` succeeds.
    """

    partition = data.policy_selection
    probabilities = portable_probabilities(model, partition.features)
    diagnostics = boundary_diagnostics(probabilities, model.decision_threshold)
    nearest = diagnostics["nearest_observed_value_below_threshold"]
    if nearest is None:
        tied_attack = 0
        tied_legitimate = 0
    else:
        tied = probabilities == nearest
        tied_attack = int(np.count_nonzero(tied & (partition.targets == 1)))
        tied_legitimate = int(np.count_nonzero(tied & (partition.targets == 0)))
    return {
        "decision_threshold": model.decision_threshold,
        "distinct_probability_count": diagnostics["distinct_probability_count"],
        "nearest_observed_probability_below_threshold": nearest,
        "note": DECISION_BOUNDARY_NOTE,
        "partition": "policy_selection",
        "rows_tied_at_that_value": {
            "attack": tied_attack,
            "legitimate": tied_legitimate,
            "total": diagnostics["tied_rows_at_nearest_value_below_threshold"],
        },
        "threshold_is_an_observed_value": diagnostics["threshold_is_an_observed_value"],
    }


def train_model(
    simulation_directory: Path,
    feature_directory: Path,
    config_path: Path,
    output_directory: Path,
) -> PublicationResult:
    require_output_separate(output_directory, (simulation_directory, feature_directory))
    config = load_modeling_config(config_path)
    loaded = load_training_data(
        simulation_directory, feature_directory, config, include_held_out_sample=False
    )
    if not isinstance(loaded, TrainingData):
        raise ModelingTrainingError("training_data_mode_invalid")
    outputs = build_training_outputs(loaded, config)
    return publish_model(
        output_directory,
        outputs.model,
        outputs.report,
        config,
        runtime_dependency_versions(),
    )


def validate_model(
    simulation_directory: Path,
    feature_directory: Path,
    config_path: Path,
    model_directory: Path,
) -> dict[str, Any]:
    config = load_modeling_config(config_path)
    locked_model, locked_report, _ = load_locked_model(model_directory, config)
    loaded = load_training_data(
        simulation_directory, feature_directory, config, include_held_out_sample=True
    )
    if not isinstance(loaded, ValidationData):
        raise ModelingTrainingError("validation_data_mode_invalid")
    outputs = build_training_outputs(loaded.training, config)
    if canonical_json_bytes(outputs.model.model_dump(mode="json")) != canonical_json_bytes(
        locked_model.model_dump(mode="json")
    ):
        raise ModelingTrainingError("deterministic_model_retraining_mismatch")
    if canonical_json_bytes(outputs.report) != canonical_json_bytes(locked_report):
        raise ModelingTrainingError("deterministic_report_retraining_mismatch")
    parity = {
        "calibration_fit": parity_result(
            outputs.winner,
            locked_model,
            _sample(loaded.training.calibration_fit.features, config.parity_sample_rows),
            config.parity_absolute_tolerance,
        ),
        "held_out_features": parity_result(
            outputs.winner,
            locked_model,
            loaded.held_out_feature_sample,
            config.parity_absolute_tolerance,
        ),
        "policy_selection": parity_result(
            outputs.winner,
            locked_model,
            _sample(loaded.training.policy_selection.features, config.parity_sample_rows),
            config.parity_absolute_tolerance,
        ),
        "train": parity_result(
            outputs.winner,
            locked_model,
            _sample(loaded.training.train.features, config.parity_sample_rows),
            config.parity_absolute_tolerance,
        ),
    }
    return {
        "decision_boundary": decision_boundary_diagnostic(locked_model, loaded.training),
        "model_id": locked_model.model_id,
        "parity": parity,
        "status": "valid",
    }
