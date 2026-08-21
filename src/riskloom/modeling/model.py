import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import GradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES
from riskloom.modeling.config import ModelingConfig


class ModelingFitError(ValueError):
    """A safe candidate-fit or portable-model error."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        allow_inf_nan=False,
    )


class PlattModel(_StrictModel):
    coefficient: float
    intercept: float
    probability_clip_epsilon: float = Field(gt=0, lt=0.01)


class LogisticPortableModel(_StrictModel):
    candidate_name: Literal["logistic_regression"]
    coefficients: list[float]
    intercept: float
    scaler_mean: list[float]
    scaler_scale: list[float]

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if not all(
            len(values) == FEATURE_COUNT
            for values in (self.coefficients, self.scaler_mean, self.scaler_scale)
        ):
            raise ValueError("portable logistic feature dimension invalid")
        if any(scale <= 0 for scale in self.scaler_scale):
            raise ValueError("portable logistic scale invalid")
        return self


class PortableTree(_StrictModel):
    node_count: int = Field(ge=1, le=100_000)
    children_left: list[int]
    children_right: list[int]
    features: list[int]
    thresholds: list[float]
    values: list[float]

    @model_validator(mode="after")
    def validate_tree(self) -> Self:
        size = len(self.values)
        if size != self.node_count or any(
            len(values) != size
            for values in (
                self.children_left,
                self.children_right,
                self.features,
                self.thresholds,
            )
        ):
            raise ValueError("portable tree array lengths invalid")
        visited: set[int] = set()
        active: set[int] = set()
        stack = [(0, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                active.remove(node)
                visited.add(node)
                continue
            if node < 0 or node >= size or node in active:
                raise ValueError("portable tree topology invalid")
            if node in visited:
                raise ValueError("portable tree node reused")
            active.add(node)
            left = self.children_left[node]
            right = self.children_right[node]
            feature = self.features[node]
            stack.append((node, True))
            if left == -1 and right == -1:
                if feature != -2:
                    raise ValueError("portable tree leaf marker invalid")
                continue
            if left == -1 or right == -1 or left == right or not 0 <= feature < FEATURE_COUNT:
                raise ValueError("portable tree split invalid")
            stack.append((right, False))
            stack.append((left, False))
        if len(visited) != size:
            raise ValueError("portable tree contains unreachable nodes")
        return self


class GradientBoostingPortableModel(_StrictModel):
    candidate_name: Literal["gradient_boosting"]
    initial_raw_score: float = Field(default=0.0, ge=0.0, le=0.0)
    learning_rate: float = Field(gt=0)
    trees: list[PortableTree]

    @model_validator(mode="after")
    def require_trees(self) -> Self:
        if not self.trees:
            raise ValueError("portable gradient boosting requires trees")
        return self


PortableCandidate = Annotated[
    LogisticPortableModel | GradientBoostingPortableModel,
    Field(discriminator="candidate_name"),
]


class LockedModel(_StrictModel):
    artifact_type: Literal["locked_binary_risk_model"] = "locked_binary_risk_model"
    product: Literal["RiskLoom"] = "RiskLoom"
    model_schema_version: Literal["1.0.0"] = "1.0.0"
    model_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_order: list[str]
    class_order: list[Literal[0, 1]]
    decision_threshold: float = Field(ge=0, le=1)
    candidate: PortableCandidate
    calibration: PlattModel

    @model_validator(mode="after")
    def validate_locked_contract(self) -> Self:
        if self.feature_order != list(FEATURE_NAMES):
            raise ValueError("portable feature order invalid")
        if self.class_order != [0, 1]:
            raise ValueError("portable class order invalid")
        return self


@dataclass(slots=True)
class FittedCandidate:
    name: Literal["logistic_regression", "gradient_boosting"]
    estimator: LogisticRegression | GradientBoostingClassifier
    scaler: StandardScaler | None
    calibrator: LogisticRegression

    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(features) if self.scaler is not None else features
        return np.asarray(self.estimator.decision_function(transformed), dtype=np.float64)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        raw = self.raw_scores(features).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(raw)[:, 1], dtype=np.float64)


def balanced_sample_weights(targets: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    negative_count = int(np.count_nonzero(targets == 0))
    positive_count = int(np.count_nonzero(targets == 1))
    total = int(targets.shape[0])
    if negative_count == 0 or positive_count == 0:
        raise ModelingFitError("candidate_fit_requires_both_classes")
    weights = {
        "legitimate": total / (2 * negative_count),
        "attack": total / (2 * positive_count),
    }
    sample_weights = np.where(targets == 1, weights["attack"], weights["legitimate"])
    return np.asarray(sample_weights, dtype=np.float64), weights


def _fit_platt(
    raw_scores: np.ndarray, targets: np.ndarray, config: ModelingConfig
) -> LogisticRegression:
    settings = config.platt_calibration
    calibrator = LogisticRegression(
        C=settings.C,
        l1_ratio=0.0,
        dual=False,
        tol=settings.tolerance,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight=None,
        random_state=settings.random_state,
        solver=settings.solver,
        max_iter=settings.maximum_iterations,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )
    calibrator.fit(raw_scores.reshape(-1, 1), targets)
    return calibrator


def fit_candidates(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    config: ModelingConfig,
) -> tuple[list[FittedCandidate], dict[str, float]]:
    sample_weights, class_weights = balanced_sample_weights(train_targets)

    logistic_config = config.logistic_regression
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    logistic_train = scaler.fit_transform(train_features)
    logistic = LogisticRegression(
        C=logistic_config.C,
        l1_ratio=logistic_config.l1_ratio,
        dual=logistic_config.dual,
        tol=logistic_config.tolerance,
        fit_intercept=logistic_config.fit_intercept,
        intercept_scaling=logistic_config.intercept_scaling,
        class_weight=None,
        random_state=logistic_config.random_state,
        solver=logistic_config.solver,
        max_iter=logistic_config.maximum_iterations,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )
    logistic.fit(logistic_train, train_targets, sample_weight=sample_weights)
    logistic_raw = np.asarray(
        logistic.decision_function(scaler.transform(calibration_features)), dtype=np.float64
    )
    logistic_fitted = FittedCandidate(
        name="logistic_regression",
        estimator=logistic,
        scaler=scaler,
        calibrator=_fit_platt(logistic_raw, calibration_targets, config),
    )

    boosting_config = config.gradient_boosting
    boosting = GradientBoostingClassifier(
        loss=boosting_config.loss,
        learning_rate=boosting_config.learning_rate,
        n_estimators=boosting_config.estimator_count,
        subsample=boosting_config.subsample,
        min_samples_split=boosting_config.minimum_samples_split,
        min_samples_leaf=boosting_config.minimum_samples_leaf,
        min_weight_fraction_leaf=0.0,
        max_depth=boosting_config.maximum_depth,
        min_impurity_decrease=boosting_config.minimum_impurity_decrease,
        init="zero",
        random_state=boosting_config.random_state,
        max_features=boosting_config.maximum_features,
        verbose=0,
        max_leaf_nodes=boosting_config.maximum_leaf_nodes,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=boosting_config.tolerance,
        ccp_alpha=0.0,
    )
    boosting.fit(train_features, train_targets, sample_weight=sample_weights)
    boosting_raw = np.asarray(boosting.decision_function(calibration_features), dtype=np.float64)
    boosting_fitted = FittedCandidate(
        name="gradient_boosting",
        estimator=boosting,
        scaler=None,
        calibrator=_fit_platt(boosting_raw, calibration_targets, config),
    )
    return [logistic_fitted, boosting_fitted], class_weights


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def export_candidate(candidate: FittedCandidate) -> PortableCandidate:
    if candidate.name == "logistic_regression":
        if not isinstance(candidate.estimator, LogisticRegression) or candidate.scaler is None:
            raise ModelingFitError("logistic_candidate_state_invalid")
        return LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=_float_list(candidate.estimator.coef_[0]),
            intercept=float(candidate.estimator.intercept_[0]),
            scaler_mean=_float_list(candidate.scaler.mean_),
            scaler_scale=_float_list(candidate.scaler.scale_),
        )
    if not isinstance(candidate.estimator, GradientBoostingClassifier):
        raise ModelingFitError("boosting_candidate_state_invalid")
    trees: list[PortableTree] = []
    for estimator in candidate.estimator.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append(
            PortableTree(
                node_count=int(tree.node_count),
                children_left=[int(value) for value in tree.children_left],
                children_right=[int(value) for value in tree.children_right],
                features=[int(value) for value in tree.feature],
                thresholds=_float_list(tree.threshold),
                values=_float_list(tree.value[:, 0, 0]),
            )
        )
    return GradientBoostingPortableModel(
        candidate_name="gradient_boosting",
        initial_raw_score=0.0,
        learning_rate=float(candidate.estimator.learning_rate),
        trees=trees,
    )


def export_platt(candidate: FittedCandidate, epsilon: float) -> PlattModel:
    return PlattModel(
        coefficient=float(candidate.calibrator.coef_[0, 0]),
        intercept=float(candidate.calibrator.intercept_[0]),
        probability_clip_epsilon=epsilon,
    )


def _validate_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT or not np.isfinite(values).all():
        raise ModelingFitError("portable_feature_matrix_invalid")
    return values


def portable_raw_scores(candidate: PortableCandidate, features: np.ndarray) -> np.ndarray:
    values = _validate_features(features)
    if isinstance(candidate, LogisticPortableModel):
        scaled = (values - np.asarray(candidate.scaler_mean)) / np.asarray(candidate.scaler_scale)
        return np.asarray(
            scaled @ np.asarray(candidate.coefficients) + candidate.intercept,
            dtype=np.float64,
        )
    scores = np.full(values.shape[0], candidate.initial_raw_score, dtype=np.float64)
    for tree in candidate.trees:
        tree_values = np.empty(values.shape[0], dtype=np.float64)
        for row_index, row in enumerate(values):
            node = 0
            while tree.children_left[node] != -1:
                feature = tree.features[node]
                node = (
                    tree.children_left[node]
                    if row[feature] <= tree.thresholds[node]
                    else tree.children_right[node]
                )
            tree_values[row_index] = tree.values[node]
        scores += candidate.learning_rate * tree_values
    return scores


def portable_probabilities(model: LockedModel, features: np.ndarray) -> np.ndarray:
    raw = portable_raw_scores(model.candidate, features)
    logits = model.calibration.coefficient * raw + model.calibration.intercept
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1 / (1 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    probabilities[~positive] = negative_exp / (1 + negative_exp)
    epsilon = model.calibration.probability_clip_epsilon
    return np.clip(probabilities, epsilon, 1 - epsilon)


def parity_result(
    candidate: FittedCandidate,
    model: LockedModel,
    features: np.ndarray,
    tolerance: float,
) -> dict[str, float | int]:
    sklearn_raw = candidate.raw_scores(features)
    portable_raw = portable_raw_scores(model.candidate, features)
    sklearn_probability = candidate.probabilities(features)
    portable_probability = portable_probabilities(model, features)
    raw_error = float(np.max(np.abs(sklearn_raw - portable_raw))) if features.shape[0] else 0.0
    probability_error = (
        float(np.max(np.abs(sklearn_probability - portable_probability)))
        if features.shape[0]
        else 0.0
    )
    if (
        not math.isfinite(raw_error + probability_error)
        or max(raw_error, probability_error) > tolerance
    ):
        raise ModelingFitError("portable_model_parity_failed")
    return {
        "checked_row_count": int(features.shape[0]),
        "maximum_probability_absolute_error": probability_error,
        "maximum_raw_score_absolute_error": raw_error,
    }
