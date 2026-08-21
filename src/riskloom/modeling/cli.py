import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from riskloom.modeling.canonical import ModelingArtifactError
from riskloom.modeling.data import ModelingDataError
from riskloom.modeling.evaluation import evaluate_model
from riskloom.modeling.training import ModelingTrainingError, train_model, validate_model


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
        else:
            result = evaluate_model(
                parsed.simulation_dir,
                parsed.feature_dir,
                parsed.config,
                parsed.model_dir,
                parsed.output_dir,
            )
            response = {"evaluation_id": result.artifact_id, "status": "evaluated"}
    except (ModelingArtifactError, ModelingDataError, ModelingTrainingError, ValueError) as error:
        print(json.dumps({"error": str(error)}, separators=(",", ":"), sort_keys=True))
        return 1
    print(json.dumps(response, separators=(",", ":"), sort_keys=True))
    return 0
