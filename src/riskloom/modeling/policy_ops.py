"""Offline orchestration for the Day 5 cost-aware policy band.

This module is the boundary where ground-truth labels meet the policy library. It lives in
``riskloom.modeling`` rather than ``riskloom.policy`` on purpose: loading labels requires
``riskloom.simulation.label_schema``, and ``riskloom.policy`` is required to stay free of that
import so its isolation from the answer key is a provable property rather than a convention.
Labels enter here only to score an already-made decision, never to make one.
"""

from pathlib import Path
from typing import Any

from riskloom.modeling.artifacts import load_locked_model
from riskloom.modeling.canonical import ModelingArtifactError
from riskloom.modeling.config import ModelingConfig, load_modeling_config
from riskloom.modeling.data import (
    ModelingDataError,
    TrainingData,
    load_policy_validation_data,
    load_training_data,
)
from riskloom.modeling.model import LockedModel, portable_probabilities
from riskloom.policy.artifacts import (
    PolicyPublicationResult,
    policy_config_sha256,
    publish_band,
    publish_comparison,
    require_output_separate,
)
from riskloom.policy.bands import (
    TIE_BREAK_ORDER,
    BandPolicy,
    boundary_diagnostics,
    evaluate_band,
    evaluate_single_threshold,
    select_band,
)
from riskloom.policy.canonical import PolicyArtifactError, canonical_sha256, read_canonical_json
from riskloom.policy.comparison import build_comparison, cost_policy
from riskloom.policy.config import PolicyConfig, load_policy_config

BAND_FIT_PARTITION = "policy_selection"


class PolicyOperationError(ValueError):
    """A safe policy orchestration error."""


def _locked_band_source(config: ModelingConfig, model: LockedModel) -> dict[str, Any]:
    contract = config.source_contract
    return {
        "effective_modeling_configuration_sha256": canonical_sha256(config.model_dump(mode="json")),
        "feature_dataset_id": contract.feature_dataset_id,
        "features_sha256": contract.features_sha256,
        "fit_partition": BAND_FIT_PARTITION,
        "labels_sha256": contract.labels_sha256,
        "model_id": model.model_id,
        "simulation_dataset_id": contract.simulation_dataset_id,
    }


def fit_policy_band(
    simulation_directory: Path,
    feature_directory: Path,
    config_path: Path,
    model_directory: Path,
    policy_config_path: Path,
    output_directory: Path,
) -> PolicyPublicationResult:
    """Fit both band thresholds on the ``policy_selection`` partition and publish the band.

    ``policy_selection`` was already the partition designated in Day 4 for choosing a decision
    rule. Fitting two thresholds on it rather than one is an extension of that designated role, not
    a new source of leakage. No held-out row and no counterfactual-validation row is read here.
    """

    require_output_separate(
        output_directory, (simulation_directory, feature_directory, model_directory)
    )
    config = load_modeling_config(config_path)
    policy = load_policy_config(policy_config_path)
    model, _, _ = load_locked_model(model_directory, config)
    loaded = load_training_data(
        simulation_directory, feature_directory, config, include_held_out_sample=False
    )
    if not isinstance(loaded, TrainingData):
        raise PolicyOperationError("policy_band_training_data_mode_invalid")

    partition = loaded.policy_selection
    probabilities = portable_probabilities(model, partition.features)
    costs = cost_policy(policy)
    # The incumbent threshold is always a sweep candidate, so the band can always express the
    # existing policy exactly and can never be fitted to something worse on this partition.
    band, outcome = select_band(
        partition.targets,
        probabilities,
        costs,
        policy.threshold_grid_size,
        required=(model.decision_threshold,),
    )
    incumbent = evaluate_single_threshold(
        partition.targets, probabilities, model.decision_threshold, costs
    )
    payload = {
        "artifact_type": "cost_aware_policy_band",
        "band": {
            "lower_threshold": band.lower_threshold,
            "upper_threshold": band.upper_threshold,
        },
        "fit_partition": BAND_FIT_PARTITION,
        "fit_partition_incumbent_outcome": incumbent.as_report(),
        "fit_partition_outcome": outcome.as_report(),
        "has_review_tier": band.has_review_tier,
        "incumbent_decision_threshold": model.decision_threshold,
        "model_id": model.model_id,
        "policy_band_schema_version": "1.0.0",
        "product": "RiskLoom",
        "threshold_boundary": {
            "banded_lower": boundary_diagnostics(probabilities, band.lower_threshold),
            "banded_upper": boundary_diagnostics(probabilities, band.upper_threshold),
            "incumbent": boundary_diagnostics(probabilities, model.decision_threshold),
        },
        "tie_break_order": list(TIE_BREAK_ORDER),
    }
    return publish_band(output_directory, payload, policy, _locked_band_source(config, model))


def load_locked_band(
    band_directory: Path, policy: PolicyConfig
) -> tuple[BandPolicy, dict[str, Any]]:
    """Load and integrity-check a published band before it is used for anything."""

    try:
        existing = {path.name for path in band_directory.iterdir()}
    except OSError:
        raise PolicyArtifactError("policy_band_directory_unreadable") from None
    if existing != {"band.json", "manifest.json"}:
        raise PolicyArtifactError("policy_band_artifact_set_invalid")
    band_value = read_canonical_json(band_directory / "band.json")
    manifest = read_canonical_json(band_directory / "manifest.json")
    if manifest.get("product") != "RiskLoom" or manifest.get("artifact_type") != (
        "cost_aware_policy_band"
    ):
        raise PolicyArtifactError("policy_band_manifest_marker_invalid")
    if manifest.get("effective_policy_configuration") != policy.model_dump(mode="json"):
        raise PolicyArtifactError("policy_band_configuration_mismatch")
    if manifest.get("effective_policy_configuration_sha256") != policy_config_sha256(policy):
        raise PolicyArtifactError("policy_band_configuration_hash_mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"band.json"}:
        raise PolicyArtifactError("policy_band_manifest_artifacts_invalid")
    entry = artifacts.get("band.json")
    if not isinstance(entry, dict) or entry.get("sha256") != canonical_sha256(band_value):
        raise PolicyArtifactError("policy_band_artifact_hash_mismatch")
    band_field = band_value.get("band")
    if not isinstance(band_field, dict):
        raise PolicyArtifactError("policy_band_schema_invalid")
    try:
        band = BandPolicy.model_validate(band_field, strict=True)
    except ValueError:
        raise PolicyArtifactError("policy_band_schema_invalid") from None
    return band, band_value


def validate_policy(
    band_directory: Path,
    validation_simulation_directory: Path,
    validation_feature_directory: Path,
    config_path: Path,
    model_directory: Path,
    policy_config_path: Path,
    output_directory: Path,
    *,
    approve: bool,
) -> tuple[PolicyPublicationResult, dict[str, Any]]:
    """Score both policies on a fresh batch and publish an honest comparison.

    The validation batch is read and never fitted against. Approval is never automatic: it requires
    an explicit human flag, and even then it is refused when any gate fails.
    """

    require_output_separate(
        output_directory,
        (
            band_directory,
            validation_simulation_directory,
            validation_feature_directory,
            model_directory,
        ),
    )
    config = load_modeling_config(config_path)
    policy = load_policy_config(policy_config_path)
    model, _, _ = load_locked_model(model_directory, config)
    band, band_value = load_locked_band(band_directory, policy)
    if band_value.get("model_id") != model.model_id:
        raise PolicyOperationError("policy_band_model_mismatch")

    data = load_policy_validation_data(
        validation_simulation_directory, validation_feature_directory, config.source_contract
    )
    probabilities = portable_probabilities(model, data.features)
    costs = cost_policy(policy)
    incumbent = evaluate_single_threshold(
        data.targets, probabilities, model.decision_threshold, costs
    )
    banded = evaluate_band(data.targets, probabilities, band, costs)
    comparison = build_comparison(
        band, model.decision_threshold, incumbent, banded, policy, probabilities
    )

    gates = comparison["gates"]
    eligible = bool(gates["approval_eligible"])
    granted = approve and eligible
    refusal_reasons: list[str] = []
    if approve and not eligible:
        refusal_reasons = list(gates["failed_gates"])
    comparison = {
        **comparison,
        "approval": {
            "approval_requested": approve,
            "approval_granted": granted,
            "note": (
                "Approval is never automatic. It requires an explicit human flag and is refused"
                " whenever any gate fails."
            ),
            "refusal_reasons": refusal_reasons,
        },
        "artifact_type": "policy_counterfactual_comparison",
        "policy_comparison_schema_version": "1.0.0",
        "product": "RiskLoom",
        "validation_batch": {
            "configuration_sha256": data.configuration_sha256,
            "dataset_profile": "policy-validation",
            "events_sha256": data.events_sha256,
            "feature_dataset_id": data.feature_dataset_id,
            "features_sha256": data.features_sha256,
            "labels_sha256": data.labels_sha256,
            "row_count": data.row_count,
            "simulation_dataset_id": data.simulation_dataset_id,
        },
    }
    source = {
        "band_id": band_value.get("band_id"),
        "model_id": model.model_id,
        "validation_feature_dataset_id": data.feature_dataset_id,
        "validation_features_sha256": data.features_sha256,
        "validation_labels_sha256": data.labels_sha256,
        "validation_simulation_dataset_id": data.simulation_dataset_id,
    }
    result = publish_comparison(output_directory, comparison, policy, source)
    return result, comparison


POLICY_ERRORS = (
    ModelingArtifactError,
    ModelingDataError,
    PolicyArtifactError,
    PolicyOperationError,
)
