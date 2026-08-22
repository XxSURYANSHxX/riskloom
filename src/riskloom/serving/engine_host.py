"""A single long-lived FeatureEngine driven one live event at a time.

The engine itself is the unmodified Day 3 class. This module only owns it, serialises access to
it, and adapts a preflight request into the exact ``CheckoutAttemptEvent`` the offline batch path
feeds it. There is no second implementation of any feature.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine
from riskloom.features.schema import FeatureRecord
from riskloom.serving.schemas import CheckoutPreflightRequest
from riskloom.simulation.event_schema import CheckoutAttemptEvent, Outcome

MILLISECOND = timedelta(milliseconds=1)


@dataclass(frozen=True, slots=True)
class ScoredEvent:
    record: FeatureRecord
    occurred_at: datetime


def preflight_to_event(
    request: CheckoutPreflightRequest, occurred_at: datetime
) -> CheckoutAttemptEvent:
    """Adapt a preflight request into the schema the feature engine already consumes.

    ``outcome`` is set to AUTHORIZED because at preflight no failure has occurred yet. This is
    sound for scoring: ``outcome`` is read exactly once in the whole engine, inside the state
    update that shapes *future* events, and never during feature computation. The consequence is a
    disclosed limitation rather than a hidden one -- see the known-limitation note in README.md.
    """

    return CheckoutAttemptEvent(
        event_id=request.event_id,
        merchant_id=request.merchant_id,
        occurred_at=occurred_at,
        checkout_id=request.checkout_id,
        customer_token=request.customer_token,
        device_token=request.device_token,
        network_token=request.network_token,
        session_token=request.session_token,
        payment_instrument_token=request.payment_instrument_token,
        amount_subunits=request.amount_subunits,
        currency=request.currency,
        outcome=Outcome.AUTHORIZED,
        failure_category=None,
        channel=request.channel,
    )


class OnlineFeatureEngine:
    """Serialised, in-process host for one warm FeatureEngine.

    A single process-wide lock is used rather than per-merchant locks. Per-merchant locking would
    be unsound here: the device, network and instrument indices are shared across merchants, and
    ``instrument_distinct_merchants_*`` is a cross-merchant feature by definition, so two merchants
    genuinely contend for the same mutable aggregates. At demo scale the critical section is
    sub-millisecond and holds no I/O, so a single lock costs nothing and is trivially correct.

    State is in-memory only. It does not survive a process restart; after a restart the engine is
    cold and history features read zero until live traffic rebuilds them. This is a stated
    limitation of this build, not an oversight. Warm-starting from the decision ledger is future
    work.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self._engine = FeatureEngine(config)
        self._lock = asyncio.Lock()
        self._last_occurred_at: datetime | None = None

    def _next_timestamp(self) -> datetime:
        """Millisecond-precision UTC, strictly greater than the previous event's timestamp.

        The engine rejects input that is not strictly increasing. Deriving the timestamp here,
        under the lock, guarantees that rejection can never be triggered by clock granularity, a
        clock stepping backwards, or two requests landing in the same millisecond.
        """

        sampled = datetime.now(UTC)
        now = sampled.replace(microsecond=(sampled.microsecond // 1_000) * 1_000)
        if self._last_occurred_at is not None and now <= self._last_occurred_at:
            now = self._last_occurred_at + MILLISECOND
        self._last_occurred_at = now
        return now

    async def process(self, request: CheckoutPreflightRequest) -> ScoredEvent:
        """Advance state by exactly one event and return its features.

        The lock covers timestamp assignment and the engine call only. Order creation and database
        writes happen outside it so that network and storage latency never serialise the engine.
        """

        async with self._lock:
            occurred_at = self._next_timestamp()
            event = preflight_to_event(request, occurred_at)
            record = self._engine.process(event)
        return ScoredEvent(record=record, occurred_at=occurred_at)

    def diagnostics(self) -> dict[str, object]:
        """Engine state summary. Never returned over the API."""

        return self._engine.diagnostics()
