from pathlib import Path
from typing import Any

from riskloom.modeling.artifacts import (
    PublicationResult,
    load_locked_model,
    publish_evaluation,
    require_output_separate,
)
from riskloom.modeling.canonical import file_metadata
from riskloom.modeling.config import load_modeling_config
from riskloom.modeling.data import EvaluationData, load_evaluation_data
from riskloom.modeling.metrics import complete_metrics
from riskloom.modeling.model import LockedModel, portable_probabilities


def build_evaluation_report(
    model: LockedModel,
    data: EvaluationData,
    *,
    false_positive_cost: int,
    false_negative_cost: int,
    reliability_bin_count: int,
) -> dict[str, Any]:
    probabilities = portable_probabilities(model, data.features)
    return {
        "artifact_type": "held_out_model_evaluation",
        "evaluation_schema_version": "1.0.0",
        "metrics": complete_metrics(
            data.targets,
            probabilities,
            model.decision_threshold,
            false_positive_cost,
            false_negative_cost,
            reliability_bin_count,
            data.scenarios,
            data.campaign_ids,
            data.occurred_at_ms,
        ),
        "model_id": model.model_id,
        "product": "RiskLoom",
        "row_count": int(data.targets.shape[0]),
    }


def evaluate_model(
    simulation_directory: Path,
    feature_directory: Path,
    config_path: Path,
    model_directory: Path,
    output_directory: Path,
) -> PublicationResult:
    require_output_separate(
        output_directory, (simulation_directory, feature_directory, model_directory)
    )
    config = load_modeling_config(config_path)
    model, _, manifest = load_locked_model(model_directory, config)
    data = load_evaluation_data(simulation_directory, feature_directory, config)
    evaluation = build_evaluation_report(
        model,
        data,
        false_positive_cost=config.false_positive_cost_units,
        false_negative_cost=config.false_negative_cost_units,
        reliability_bin_count=config.reliability_bin_count,
    )
    artifacts = manifest["artifacts"]
    locked_model_sha256 = artifacts["model.json"]["sha256"]
    locked_model_manifest_sha256 = file_metadata(model_directory / "manifest.json")["sha256"]
    return publish_evaluation(
        output_directory,
        evaluation,
        model,
        config,
        str(locked_model_sha256),
        str(locked_model_manifest_sha256),
    )
