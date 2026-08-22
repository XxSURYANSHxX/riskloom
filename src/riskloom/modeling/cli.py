import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from riskloom.modeling.canonical import ModelingArtifactError
from riskloom.modeling.data import ModelingDataError
from riskloom.modeling.evaluation import evaluate_model
from riskloom.modeling.policy_ops import PolicyOperationError, fit_policy_band, validate_policy
from riskloom.modeling.training import ModelingTrainingError, train_model, validate_model
from riskloom.policy.canonical import PolicyArtifactError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m riskloom.modeling")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "validate-model", "evaluate-test"):
        command = commands.add_parser(name)
        command.add_argument("--simulation-dir", type=Path, required=True)
        command.add_argument("--feature-dir", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--model-dir", type=Path, required=name != "train")
        command.add_argument("--output-dir", type=Path, required=name != "validate-model")

    fit_band = commands.add_parser("fit-policy-band")
    fit_band.add_argument("--simulation-dir", type=Path, required=True)
    fit_band.add_argument("--feature-dir", type=Path, required=True)
    fit_band.add_argument("--config", type=Path, required=True)
    fit_band.add_argument("--model-dir", type=Path, required=True)
    fit_band.add_argument("--policy-config", type=Path, required=True)
    fit_band.add_argument("--output-dir", type=Path, required=True)

    validate_band = commands.add_parser("validate-policy")
    validate_band.add_argument("--band-dir", type=Path, required=True)
    validate_band.add_argument("--validation-simulation-dir", type=Path, required=True)
    validate_band.add_argument("--validation-feature-dir", type=Path, required=True)
    validate_band.add_argument("--config", type=Path, required=True)
    validate_band.add_argument("--model-dir", type=Path, required=True)
    validate_band.add_argument("--policy-config", type=Path, required=True)
    validate_band.add_argument("--output-dir", type=Path, required=True)
    # Explicit human approval. Never implied, never defaulted on, and refused outright when any
    # gate in the published comparison failed.
    validate_band.add_argument("--approve", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "train":
            result = train_model(
                parsed.simulation_dir, parsed.feature_dir, parsed.config, parsed.output_dir
            )
            response = {"model_id": result.artifact_id, "status": "trained"}
        elif parsed.command == "validate-model":
            response = validate_model(
                parsed.simulation_dir, parsed.feature_dir, parsed.config, parsed.model_dir
            )
        elif parsed.command == "fit-policy-band":
            band_result = fit_policy_band(
                parsed.simulation_dir,
                parsed.feature_dir,
                parsed.config,
                parsed.model_dir,
                parsed.policy_config,
                parsed.output_dir,
            )
            response = {"band_id": band_result.artifact_id, "status": "band_fitted"}
        elif parsed.command == "validate-policy":
            comparison_result, comparison = validate_policy(
                parsed.band_dir,
                parsed.validation_simulation_dir,
                parsed.validation_feature_dir,
                parsed.config,
                parsed.model_dir,
                parsed.policy_config,
                parsed.output_dir,
                approve=parsed.approve,
            )
            gates = comparison["gates"]
            approval = comparison["approval"]
            response = {
                "approval_eligible": gates["approval_eligible"],
                "approval_granted": approval["approval_granted"],
                "beats_incumbent_cost": gates["beats_incumbent_cost"],
                "comparison_id": comparison_result.artifact_id,
                "cost_delta_units": gates["cost_delta_units"],
                "failed_gates": gates["failed_gates"],
                "status": "validated",
            }
            # The report is published either way, because a policy that loses is a result worth
            # recording. Requesting approval that the evidence does not support is still a failure.
            if parsed.approve and not approval["approval_granted"]:
                print(json.dumps(response, separators=(",", ":"), sort_keys=True))
                return 1
        else:
            result = evaluate_model(
                parsed.simulation_dir,
                parsed.feature_dir,
                parsed.config,
                parsed.model_dir,
                parsed.output_dir,
            )
            response = {"evaluation_id": result.artifact_id, "status": "evaluated"}
    except (
        ModelingArtifactError,
        ModelingDataError,
        ModelingTrainingError,
        PolicyArtifactError,
        PolicyOperationError,
        ValueError,
    ) as error:
        print(json.dumps({"error": str(error)}, separators=(",", ":"), sort_keys=True))
        return 1
    print(json.dumps(response, separators=(",", ":"), sort_keys=True))
    return 0
