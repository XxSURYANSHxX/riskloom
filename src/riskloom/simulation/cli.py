from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from riskloom.simulation.event_schema import CheckoutAttemptEvent
from riskloom.simulation.replay import ReplayOptions, ReplayResult, replay_jsonl

if TYPE_CHECKING:
    from riskloom.simulation.artifacts import GenerationResult


class CountingConsumer:
    async def consume(self, event: CheckoutAttemptEvent) -> None:
        del event


def _generation_output(result: GenerationResult) -> dict[str, Any]:
    return {
        "artifact_hashes": {
            key: result.artifact_hashes[key] for key in sorted(result.artifact_hashes)
        },
        "dataset_id": result.dataset_id,
        "event_count": result.event_count,
        "output_directory": str(result.output_directory),
        "status": "generated",
    }


def _replay_output(result: ReplayResult) -> dict[str, Any]:
    return {
        "events_emitted": result.events_emitted,
        "first_occurred_at": result.first_occurred_at.isoformat()
        if result.first_occurred_at
        else None,
        "last_occurred_at": result.last_occurred_at.isoformat()
        if result.last_occurred_at
        else None,
        "status": "replayed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RiskLoom local synthetic simulation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate validated synthetic artifacts")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")

    replay = subparsers.add_parser("replay", help="replay model-visible events in process")
    replay.add_argument("--events", type=Path, required=True)
    replay.add_argument("--timing", choices=("no-delay", "scaled"), default="no-delay")
    replay.add_argument("--speed-factor", type=int, default=1)
    replay.add_argument("--maximum-events", type=int)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "generate":
        from riskloom.simulation.artifacts import generate_dataset
        from riskloom.simulation.config import load_generator_config

        config = load_generator_config(arguments.config)
        result = generate_dataset(
            config,
            arguments.seed,
            arguments.output_dir,
            overwrite=arguments.overwrite,
        )
        output = _generation_output(result)
    else:
        options = ReplayOptions(
            timing="no_delay" if arguments.timing == "no-delay" else "scaled",
            speed_factor=arguments.speed_factor,
            maximum_events=arguments.maximum_events,
        )
        replay_result = asyncio.run(replay_jsonl(arguments.events, CountingConsumer(), options))
        output = _replay_output(replay_result)
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
