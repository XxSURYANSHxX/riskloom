import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from riskloom.features.config import FeatureConfig
from riskloom.simulation.event_schema import (
    Channel,
    CheckoutAttemptEvent,
    FailureCategory,
    Outcome,
)

EventFactory = Callable[..., CheckoutAttemptEvent]


@pytest.fixture
def feature_config() -> FeatureConfig:
    return FeatureConfig.model_validate(
        {
            "config_schema_version": "1.0.0",
            "feature_schema_version": "1.0.0",
            "rolling_windows_seconds": [60, 300, 3600],
            "checkout_history_window_seconds": 3600,
            "failure_rate_window_seconds": 300,
        }
    )


@pytest.fixture
def event_factory() -> EventFactory:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def build(
        index: int,
        *,
        seconds: int = 0,
        milliseconds: int = 0,
        outcome: Outcome = Outcome.AUTHORIZED,
        **overrides: Any,
    ) -> CheckoutAttemptEvent:
        suffix = f"{index:032x}"
        values: dict[str, Any] = {
            "event_id": f"evt_{suffix}",
            "merchant_id": f"mrc_{'1' * 32}",
            "occurred_at": start + timedelta(seconds=seconds, milliseconds=milliseconds),
            "checkout_id": f"chk_{'2' * 32}",
            "customer_token": f"cus_{'3' * 32}",
            "device_token": f"dev_{'4' * 32}",
            "network_token": f"net_{'5' * 32}",
            "session_token": f"ses_{'6' * 32}",
            "payment_instrument_token": f"pmt_{'7' * 32}",
            "amount_subunits": 10_000 + index,
            "currency": "INR",
            "outcome": outcome,
            "failure_category": (
                FailureCategory.INSTRUMENT_DECLINED if outcome is Outcome.FAILED else None
            ),
            "channel": Channel.WEB,
        }
        values.update(overrides)
        return CheckoutAttemptEvent.model_validate(values)

    return build


@pytest.fixture
def write_events() -> Callable[[Path, Sequence[CheckoutAttemptEvent]], Path]:
    def write(path: Path, events: Sequence[CheckoutAttemptEvent]) -> Path:
        rows = (
            json.dumps(
                event.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for event in events
        )
        path.write_bytes("".join(rows).encode("utf-8"))
        return path

    return write


@pytest.fixture
def tiny_events(event_factory: EventFactory) -> list[CheckoutAttemptEvent]:
    return [
        event_factory(
            index,
            seconds=index * 7,
            outcome=Outcome.FAILED if index % 4 == 0 else Outcome.AUTHORIZED,
            channel=(Channel.WEB, Channel.MOBILE_WEB, Channel.MOBILE_APP)[index % 3],
        )
        for index in range(1, 13)
    ]


@pytest.fixture
def tiny_events_path(
    tmp_path: Path,
    tiny_events: list[CheckoutAttemptEvent],
    write_events: Callable[[Path, Sequence[CheckoutAttemptEvent]], Path],
) -> Path:
    return write_events(tmp_path / "events.jsonl", tiny_events)
