import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from riskloom.simulation.artifacts import write_event_jsonl
from riskloom.simulation.event_schema import CheckoutAttemptEvent
from riskloom.simulation.generation import GeneratedRecord
from riskloom.simulation.replay import (
    ReplayConsumerError,
    ReplayInputError,
    ReplayOptions,
    replay_jsonl,
)


class CollectingConsumer:
    def __init__(self, fail_after: int | None = None) -> None:
        self.events: list[CheckoutAttemptEvent] = []
        self.fail_after = fail_after

    async def consume(self, event: CheckoutAttemptEvent) -> None:
        if self.fail_after is not None and len(self.events) >= self.fail_after:
            raise RuntimeError("synthetic consumer failure")
        self.events.append(event)


@pytest.mark.asyncio
async def test_no_delay_replay_is_ordered_and_label_blind(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    consumer = CollectingConsumer()
    result = await replay_jsonl(output / "events.jsonl", consumer, ReplayOptions())
    assert result.events_emitted == 300
    assert [event.event_id for event in consumer.events] == sorted(
        (event.event_id for event in consumer.events),
        key=lambda event_id: next(
            (event.occurred_at, event.event_id)
            for event in consumer.events
            if event.event_id == event_id
        ),
    )


@pytest.mark.asyncio
async def test_scaled_replay_uses_fake_clock(
    tiny_records: list[GeneratedRecord], tmp_path: Path
) -> None:
    events = [tiny_records[0].event, tiny_records[1].event, tiny_records[2].event]
    path = tmp_path / "events.jsonl"
    write_event_jsonl(path, events)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    with patch("riskloom.simulation.replay.asyncio.sleep", new=fake_sleep):
        result = await replay_jsonl(
            path,
            CollectingConsumer(),
            ReplayOptions(timing="scaled", speed_factor=100),
        )
    assert result.events_emitted == 3
    expected = [
        (events[index].occurred_at - events[index - 1].occurred_at).total_seconds() / 100
        for index in range(1, len(events))
    ]
    assert sleeps == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["malformed", "duplicate", "unsorted", "blank"])
async def test_malformed_inputs_fail_safely(
    kind: str,
    tiny_records: list[GeneratedRecord],
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    first = tiny_records[0].event
    second = tiny_records[1].event
    if kind == "malformed":
        path.write_bytes(b"{not-json\n")
    elif kind == "duplicate":
        write_event_jsonl(path, [first, first])
    elif kind == "unsorted":
        write_event_jsonl(path, [second, first])
    else:
        path.write_bytes(b"\n")
    with pytest.raises(ReplayInputError) as exc_info:
        await replay_jsonl(path, CollectingConsumer(), ReplayOptions())
    assert "not-json" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_consumer_failure_stops_without_retry(
    tiny_output: tuple[Path, object],
) -> None:
    output, _ = tiny_output
    consumer = CollectingConsumer(fail_after=1)
    with pytest.raises(ReplayConsumerError):
        await replay_jsonl(output / "events.jsonl", consumer, ReplayOptions())
    assert len(consumer.events) == 1


def test_replay_options_are_bounded() -> None:
    with pytest.raises(ValueError):
        ReplayOptions(timing="scaled", speed_factor=3_601)
    with pytest.raises(ValueError):
        ReplayOptions(speed_factor=2)
    with pytest.raises(ValueError):
        ReplayOptions(maximum_events=0)
    with pytest.raises(ValueError):
        ReplayOptions(timing="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReplayOptions(speed_factor=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReplayOptions(maximum_events=True)  # type: ignore[arg-type]


def test_replay_module_has_no_label_or_network_surface() -> None:
    source_root = Path(__file__).parents[3] / "src/riskloom/simulation"
    source_path = source_root / "replay.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for module in (source_path, source_root / "event_schema.py"):
        module_tree = ast.parse(module.read_text(encoding="utf-8"))
        imports.update(
            alias.name
            for node in ast.walk(module_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        imports.update(
            node.module or "" for node in ast.walk(module_tree) if isinstance(node, ast.ImportFrom)
        )
    assert not any(
        forbidden in imported.casefold()
        for imported in imports
        for forbidden in ("label", "http", "socket", "razorpay")
    )
    replay_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "replay_jsonl"
    )
    argument_names = {
        argument.arg for argument in (*replay_function.args.args, *replay_function.args.kwonlyargs)
    }
    assert not {"label", "target", "url"}.intersection(argument_names)
    assert argument_names == {"path", "consumer", "options"}
    source = source_path.read_text(encoding="utf-8").casefold()
    assert not any(
        forbidden in source
        for forbidden in ("__import__", "eval(", "exec(", "subprocess", "labels.jsonl")
    )

    cli_tree = ast.parse((source_root / "cli.py").read_text(encoding="utf-8"))
    consumer = next(
        node
        for node in cli_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CountingConsumer"
    )
    consume = next(
        node
        for node in consumer.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "consume"
    )
    assert not any(isinstance(node, ast.Call) for node in ast.walk(consume))
