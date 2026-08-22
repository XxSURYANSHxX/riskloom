"""Train/serve skew regression — the most important test in this gate.

The guarantee: a sequence of events scored one at a time through the live online path produces
feature vectors byte-identical to the same sequence processed through the real offline batch
extraction path. Not a hand-rolled loop standing in for the batch path -- this drives
``extract_feature_dataset``, the same function that produced the training data the locked model
was fitted on.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from riskloom.features.artifacts import canonical_json_bytes
from riskloom.features.config import FeatureConfig, load_feature_config
from riskloom.features.extraction import extract_feature_dataset
from riskloom.features.schema import FEATURE_NAMES, FeatureRecord
from riskloom.serving.engine_host import OnlineFeatureEngine
from riskloom.serving.schemas import CheckoutPreflightRequest
from riskloom.simulation.event_schema import (
    Channel,
    CheckoutAttemptEvent,
    FailureCategory,
    Outcome,
)


@pytest.fixture(scope="module")
def feature_config() -> FeatureConfig:
    return load_feature_config(Path("configs/features/default.json"))


def _token(prefix: str, index: int) -> str:
    return f"{prefix}_{index:032x}"


def _requests(count: int) -> list[CheckoutPreflightRequest]:
    """A sequence with deliberate entity reuse so history features are genuinely exercised."""

    channels = (Channel.WEB, Channel.MOBILE_WEB, Channel.MOBILE_APP)
    built: list[CheckoutPreflightRequest] = []
    for index in range(count):
        built.append(
            CheckoutPreflightRequest(
                event_id=_token("evt", index + 1),
                merchant_id=_token("mrc", index % 3),
                checkout_id=_token("chk", index % 7),
                customer_token=_token("cus", index % 5) if index % 4 else None,
                device_token=_token("dev", index % 4) if index % 6 else None,
                network_token=_token("net", index % 2) if index % 5 else None,
                session_token=_token("ses", index % 6),
                payment_instrument_token=_token("pmt", index % 8),
                amount_subunits=10_000 + index * 137,
                currency="INR",
                channel=channels[index % 3],
            )
        )
    return built


def _event(
    request: CheckoutPreflightRequest,
    occurred_at: datetime,
    *,
    outcome: Outcome = Outcome.AUTHORIZED,
    failure_category: FailureCategory | None = None,
) -> CheckoutAttemptEvent:
    """Build the offline event directly, deliberately NOT through the serving adapter.

    Using the adapter on both sides would make the comparison circular.
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
        outcome=outcome,
        failure_category=failure_category,
        channel=request.channel,
    )


def _run_online(
    requests: list[CheckoutPreflightRequest], config: FeatureConfig
) -> tuple[list[FeatureRecord], list[datetime]]:
    engine = OnlineFeatureEngine(config)

    async def drive() -> tuple[list[FeatureRecord], list[datetime]]:
        records: list[FeatureRecord] = []
        stamps: list[datetime] = []
        for request in requests:
            scored = await engine.process(request)
            records.append(scored.record)
            stamps.append(scored.occurred_at)
        return records, stamps

    return asyncio.run(drive())


def _run_offline(
    events: list[CheckoutAttemptEvent], config: FeatureConfig, tmp_path: Path
) -> list[FeatureRecord]:
    """Drive the real offline extraction pipeline end to end."""

    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(
        b"".join(canonical_json_bytes(event.model_dump(mode="json")) for event in events)
    )
    output = tmp_path / "features"
    extract_feature_dataset(events_path, config, output)
    return [
        FeatureRecord.model_validate(json.loads(line), strict=False)
        for line in (output / "features.jsonl").read_bytes().splitlines()
        if line.strip()
    ]


def test_online_and_offline_paths_produce_identical_feature_vectors(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    """All 75 features identical, for every event, across both paths.

    Timestamps are taken from the online run and replayed offline, so the comparison isolates
    feature logic from wall-clock scheduling. Everything else about the two paths is independent.
    """

    requests = _requests(60)
    online_records, stamps = _run_online(requests, feature_config)
    events = [_event(request, stamp) for request, stamp in zip(requests, stamps, strict=True)]
    offline_records = _run_offline(events, feature_config, tmp_path)

    assert len(offline_records) == len(online_records) == 60
    for index, (online, offline) in enumerate(zip(online_records, offline_records, strict=True)):
        assert online.event_id == offline.event_id, index
        assert online.occurred_at == offline.occurred_at, index
        online_values = online.features.model_dump()
        offline_values = offline.features.model_dump()
        for name in FEATURE_NAMES:
            assert online_values[name] == offline_values[name], (index, name)
        assert online.features == offline.features, index


def test_history_features_are_actually_exercised_by_the_parity_fixture(
    feature_config: FeatureConfig,
) -> None:
    """Guard against the parity test passing trivially on an all-zero feature matrix."""

    records, _ = _run_online(_requests(60), feature_config)
    final = records[-1].features.model_dump()
    assert final["merchant_prior_attempt_count_3600s"] > 0
    assert final["device_prior_attempt_count_3600s"] > 0
    assert final["network_prior_attempt_count_3600s"] > 0
    assert final["instrument_prior_attempt_count_3600s"] > 0
    assert final["session_prior_attempt_count_300s"] > 0
    assert final["checkout_prior_attempt_count_3600s"] > 0
    non_zero = sum(1 for name in FEATURE_NAMES if final[name] != 0)
    assert non_zero >= 30


OUTCOME_DERIVED = tuple(name for name in FEATURE_NAMES if "failure" in name)
OUTCOME_INDEPENDENT = tuple(name for name in FEATURE_NAMES if "failure" not in name)


def test_outcome_independent_features_match_even_when_true_outcomes_differ(
    tmp_path: Path, feature_config: FeatureConfig
) -> None:
    """Pin the disclosed limitation instead of hiding it.

    The online adapter records every live attempt as authorized, because at preflight no failure
    has occurred yet. When the true offline sequence contains failures, the 57 outcome-independent
    features still match exactly; the 18 failure-derived ones diverge, and that divergence is
    asserted here rather than left to be discovered later.
    """

    assert len(OUTCOME_DERIVED) == 18
    assert len(OUTCOME_INDEPENDENT) == 57

    requests = _requests(40)
    online_records, stamps = _run_online(requests, feature_config)
    # Every third event genuinely failed in the offline truth.
    events = [
        _event(
            request,
            stamp,
            outcome=Outcome.FAILED if index % 3 == 0 else Outcome.AUTHORIZED,
            failure_category=FailureCategory.INSTRUMENT_DECLINED if index % 3 == 0 else None,
        )
        for index, (request, stamp) in enumerate(zip(requests, stamps, strict=True))
    ]
    offline_records = _run_offline(events, feature_config, tmp_path)

    diverged: set[str] = set()
    for online, offline in zip(online_records, offline_records, strict=True):
        online_values = online.features.model_dump()
        offline_values = offline.features.model_dump()
        for name in OUTCOME_INDEPENDENT:
            assert online_values[name] == offline_values[name], name
        for name in OUTCOME_DERIVED:
            if online_values[name] != offline_values[name]:
                diverged.add(name)

    # The failure-derived features do diverge, and only those. If this ever becomes empty the
    # limitation has been fixed and the note in README.md should be revisited.
    assert diverged, "expected the failure-derived features to diverge; limitation may be stale"
    assert diverged.issubset(set(OUTCOME_DERIVED))


def test_online_adapter_differs_from_a_true_event_only_in_outcome_fields() -> None:
    from riskloom.serving.engine_host import preflight_to_event  # noqa: PLC0415

    request = _requests(1)[0]
    stamp = datetime.fromisoformat("2026-03-01T00:00:00+00:00")
    adapted = preflight_to_event(request, stamp).model_dump(mode="json")
    truth = _event(
        request,
        stamp,
        outcome=Outcome.FAILED,
        failure_category=FailureCategory.AUTHENTICATION,
    ).model_dump(mode="json")
    differing = {key for key in truth if truth[key] != adapted[key]}
    assert differing == {"outcome", "failure_category"}
