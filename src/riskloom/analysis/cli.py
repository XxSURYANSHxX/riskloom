"""Offline analysis commands.

Analysis is always invoked deliberately and never runs as part of serving. Nothing reachable from
here can alter a decision, a threshold, or a model.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from riskloom.analysis import adversarial_stress


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m riskloom.analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    stress = commands.add_parser(
        "adversarial-stress",
        help="score the locked model against evasion-shaped attack traffic",
    )
    stress.add_argument("--simulations-root", type=Path, default=Path("artifacts/simulations"))
    stress.add_argument("--features-root", type=Path, default=Path("artifacts/features"))
    stress.add_argument("--model-dir", type=Path, default=Path("artifacts/models/development"))
    stress.add_argument(
        "--modeling-config", type=Path, default=Path("configs/modeling/default.json")
    )
    stress.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis"))
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if parsed.command == "adversarial-stress":
        report = adversarial_stress.run(
            simulations_root=parsed.simulations_root,
            features_root=parsed.features_root,
            model_dir=parsed.model_dir,
            output_dir=parsed.output_dir,
            modeling_config_path=parsed.modeling_config,
        )
        print(
            json.dumps(
                {
                    "status": "reported",
                    "output_directory": str(parsed.output_dir),
                    "variants": [item["variant"] for item in report["variants"]],
                },
                sort_keys=True,
            )
        )
    return 0
