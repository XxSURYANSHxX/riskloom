"""Ordering, locking and state-lifecycle behaviour of the online engine host."""

import asyncio
from pathlib import Path

import pytest

from riskloom.features.config import FeatureConfig, load_feature_config
from riskloom.features.schema import FEATURE_NAMES
from riskloom.serving.engine_host import OnlineFeatureEngine
from riskloom.serving.schemas import CheckoutPreflightRequest
from riskloom.simulation.event_schema import Channel

SHARED_DEVICE = "dev_" + "a" * 32
SHARED_NETWORK = "net_" + "b" * 32


@pytest.fixture(scope="module")
def feature_config() -> FeatureConfig:
    return load_feature_config(Path("configs/features/default.json"))


def _request(index: int, *, shared: bool = True) -> CheckoutPreflightRequest:
    return CheckoutPreflightRequest(
        event_id=f"evt_{index:032x}",
        merchant_id=f"mrc_{index % 3:032x}",
        checkout_id=f"chk_{index:032x}",
        customer_token=None,
        device_token=SHARED_DEVICE if shared else f"dev_{index:032x}",
        network_token=SHARED_NETWORK if shared else f"net_{index:032x}",
        session_token=f"ses_{index:032x}",
        payment_instrument_token=f"pmt_{index:032x}",
        amount_subunits=25_000,
        currency="INR",
        channel=Channel.WEB,
    )


def test_timestamps_are_strictly_increasing_under_a_same_millisecond_burst(
    feature_config: FeatureConfig,
) -> None:
    """The engine rejects non-monotonic input; the host must make that unreachable."""

    engine = OnlineFeatureEngine(feature_config)

    async def drive() -> list[object]:
        return [(await engine.process(_request(index))).occurred_at for index in range(50)]

    stamps = asyncio.run(drive())
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 50
    for earlier, later in zip(stamps, stamps[1:], strict=False):
        assert later > earlier
    # Millisecond precision is required by CheckoutAttemptEvent.
    assert all(stamp.microsecond % 1_000 == 0 for stamp in stamps)  # type: ignore[attr-defined]


def test_concurrent_requests_do_not_lose_a_state_update(feature_config: FeatureConfig) -> None:
    """The race test.

    Every request shares one device token, so each must observe exactly the number of prior
    attempts that preceded it. Without serialisation around read-then-update, two coroutines would
    read the same prior count and the sequence would contain duplicates.
    """

    engine = OnlineFeatureEngine(feature_config)
    count = 40

    async def drive() -> list[int]:
        results = await asyncio.gather(*(engine.process(_request(index)) for index in range(count)))
        return [
            scored.record.features.model_dump()["device_prior_attempt_count_3600s"]
            for scored in results
        ]

    counts = asyncio.run(drive())
    # Each concurrent request saw a distinct, contiguous prior count: no lost update anywhere.
    assert sorted(counts) == list(range(count))


def test_concurrent_requests_keep_engine_state_internally_consistent(
    feature_config: FeatureConfig,
) -> None:
    engine = OnlineFeatureEngine(feature_config)

    async def drive() -> None:
        await asyncio.gather(*(engine.process(_request(index)) for index in range(30)))

    asyncio.run(drive())
    diagnostics = engine.diagnostics()
    assert diagnostics["maximum_window_seconds"] == 3_600


def test_a_fresh_host_starts_cold(feature_config: FeatureConfig) -> None:
    """In-memory state only: a new process begins with no history. Disclosed limitation."""

    engine = OnlineFeatureEngine(feature_config)

    async def drive() -> dict[str, int]:
        scored = await engine.process(_request(1))
        return scored.record.features.model_dump()

    values = asyncio.run(drive())
    history = [name for name in FEATURE_NAMES if "prior" in name or "distinct" in name]
    assert history
    assert all(values[name] == 0 for name in history)


def test_each_host_owns_independent_state(feature_config: FeatureConfig) -> None:
    first = OnlineFeatureEngine(feature_config)
    second = OnlineFeatureEngine(feature_config)

    async def drive() -> tuple[int, int]:
        for index in range(5):
            await first.process(_request(index))
        warm = (await first.process(_request(99))).record.features.model_dump()
        cold = (await second.process(_request(99))).record.features.model_dump()
        return (
            warm["device_prior_attempt_count_3600s"],
            cold["device_prior_attempt_count_3600s"],
        )

    warm_count, cold_count = asyncio.run(drive())
    assert warm_count == 5
    assert cold_count == 0
