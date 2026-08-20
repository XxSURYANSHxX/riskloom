import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from riskloom.simulation.event_schema import CheckoutAttemptEvent

MAXIMUM_LINE_BYTES = 1_048_576


class CheckoutAttemptConsumer(Protocol):
    async def consume(self, event: CheckoutAttemptEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    timing: Literal["no_delay", "scaled"] = "no_delay"
    speed_factor: int = 1
    maximum_events: int | None = None

    def __post_init__(self) -> None:
        if self.timing not in ("no_delay", "scaled"):
            raise ValueError("replay_timing_invalid")
        if type(self.speed_factor) is not int or not 1 <= self.speed_factor <= 3_600:
            raise ValueError("replay_speed_factor_out_of_range")
        if self.timing == "no_delay" and self.speed_factor != 1:
            raise ValueError("speed_factor_requires_scaled_timing")
        if self.maximum_events is not None and (
            type(self.maximum_events) is not int or self.maximum_events <= 0
        ):
            raise ValueError("maximum_events_must_be_positive")


@dataclass(frozen=True, slots=True)
class ReplayResult:
    events_emitted: int
    first_occurred_at: datetime | None
    last_occurred_at: datetime | None


class ReplayInputError(ValueError):
    pass


class ReplayConsumerError(RuntimeError):
    pass


async def replay_jsonl(
    path: Path,
    consumer: CheckoutAttemptConsumer,
    options: ReplayOptions,
) -> ReplayResult:
    seen_ids: set[str] = set()
    previous: CheckoutAttemptEvent | None = None
    first_at: datetime | None = None
    emitted = 0
    try:
        stream = path.open("rb")
    except OSError:
        raise ReplayInputError("replay_input_unreadable") from None

    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if options.maximum_events is not None and emitted >= options.maximum_events:
                break
            if len(raw_line) > MAXIMUM_LINE_BYTES:
                raise ReplayInputError(f"replay_line_oversized:{line_number}")
            if not raw_line.strip():
                raise ReplayInputError(f"replay_line_blank:{line_number}")
            try:
                event = CheckoutAttemptEvent.model_validate_json(raw_line)
            except (ValidationError, ValueError):
                raise ReplayInputError(f"replay_line_invalid:{line_number}") from None
            if event.event_id in seen_ids:
                raise ReplayInputError(f"replay_event_duplicate:{line_number}")
            if previous is not None:
                current_key = (event.occurred_at, event.event_id)
                previous_key = (previous.occurred_at, previous.event_id)
                if current_key < previous_key:
                    raise ReplayInputError(f"replay_input_unsorted:{line_number}")
                if options.timing == "scaled":
                    delta_seconds = (event.occurred_at - previous.occurred_at).total_seconds()
                    await asyncio.sleep(delta_seconds / options.speed_factor)
            try:
                await consumer.consume(event)
            except Exception:
                raise ReplayConsumerError(f"replay_consumer_failed:{line_number}") from None
            seen_ids.add(event.event_id)
            first_at = first_at or event.occurred_at
            previous = event
            emitted += 1

    return ReplayResult(
        events_emitted=emitted,
        first_occurred_at=first_at,
        last_occurred_at=previous.occurred_at if previous is not None else None,
    )
