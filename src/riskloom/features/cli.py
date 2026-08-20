from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from riskloom.features.artifacts import ExtractionResult, FeatureArtifactError
from riskloom.features.config import FeatureConfigurationError, load_feature_config
from riskloom.features.extraction import FeatureExtractionError, extract_feature_dataset
from riskloom.features.validation import FeatureValidationError, validate_feature_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RiskLoom causal feature artifact tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract validated causal features")
    extract.add_argument("--events", type=Path, required=True)
    extract.add_argument("--config", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="validate features against source events")
    validate.add_argument("--events", type=Path, required=True)
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--input-dir", type=Path, required=True)
    return parser


def _extraction_output(result: ExtractionResult) -> dict[str, Any]:
    return {
        "artifact_hashes": {
            name: result.artifact_hashes[name] for name in sorted(result.artifact_hashes)
        },
        "feature_count": result.feature_count,
        "feature_dataset_id": result.feature_dataset_id,
        "row_count": result.row_count,
        "status": "generated",
    }


def run(arguments: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    try:
        config = load_feature_config(parsed.config)
        if parsed.command == "extract":
            result = extract_feature_dataset(
                parsed.events,
                config,
                parsed.output_dir,
                overwrite=parsed.overwrite,
            )
            output: dict[str, Any] = _extraction_output(result)
        else:
            output = validate_feature_dataset(parsed.events, config, parsed.input_dir)
    except (
        FeatureArtifactError,
        FeatureConfigurationError,
        FeatureExtractionError,
        FeatureValidationError,
    ) as error:
        print(
            json.dumps(
                {"error": str(error), "status": "error"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> None:
    raise SystemExit(run())
