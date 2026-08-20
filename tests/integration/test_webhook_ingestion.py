import asyncio
import copy
import hashlib
import hmac
import json
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from riskloom.core.config import Settings
from riskloom.db.models import PaymentObservation, WebhookEvent
from riskloom.db.session import Database
from riskloom.integrations.razorpay.schemas import WebhookEnvelope
from riskloom.main import create_app
from riskloom.services.webhook_ingestion import ingest_webhook
from tests.conftest import SYNTHETIC_WEBHOOK_SECRET

pytestmark = pytest.mark.integration


def _raw(event: dict[str, Any]) -> bytes:
    return json.dumps(event, separators=(",", ":")).encode("utf-8")


def _signature(raw_body: bytes) -> str:
    return hmac.new(SYNTHETIC_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def _headers(raw_body: bytes, event_id: str = "event_synthetic") -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": _signature(raw_body),
        "X-Razorpay-Event-Id": event_id,
    }


async def _events_for(database_url: str, event_ids: Sequence[str]) -> list[WebhookEvent]:
    engine = create_async_engine(database_url)
    try:
        async with AsyncSession(engine) as session:
            result = await session.scalars(
                select(WebhookEvent)
                .where(WebhookEvent.provider_event_id.in_(event_ids))
                .order_by(WebhookEvent.provider_created_at)
            )
            return list(result)
    finally:
        await engine.dispose()


async def _observations_for(database_url: str, payment_id: str) -> list[PaymentObservation]:
    engine = create_async_engine(database_url)
    try:
        async with AsyncSession(engine) as session:
            result = await session.scalars(
                select(PaymentObservation)
                .where(PaymentObservation.provider_payment_id == payment_id)
                .order_by(PaymentObservation.provider_event_created_at)
            )
            return list(result)
    finally:
        await engine.dispose()


def test_valid_webhook_is_sanitized_and_duplicate_is_idempotent(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
) -> None:
    raw_body = _raw(synthetic_event)
    event_id = "event_valid_and_duplicate"

    with TestClient(create_app(integration_settings)) as client:
        first = client.post(
            "/api/v1/webhooks/razorpay", content=raw_body, headers=_headers(raw_body, event_id)
        )
        duplicate = client.post(
            "/api/v1/webhooks/razorpay", content=raw_body, headers=_headers(raw_body, event_id)
        )

    assert first.status_code == 200
    assert first.json() == {"status": "accepted", "duplicate": False}
    assert duplicate.status_code == 200
    assert duplicate.json() == {"status": "accepted", "duplicate": True}

    events = asyncio.run(
        _events_for(integration_settings.database_url.get_secret_value(), [event_id])
    )
    observations = asyncio.run(
        _observations_for(integration_settings.database_url.get_secret_value(), "pay_synthetic")
    )
    assert len(events) == 1
    assert len([row for row in observations if row.webhook_event_id == events[0].id]) == 1
    assert events[0].raw_body_sha256 == hashlib.sha256(raw_body).hexdigest()
    stored = json.dumps(events[0].sanitized_payload, sort_keys=True)
    for forbidden in (
        "person@example.invalid",
        "+10000000000",
        "192.0.2.1",
        "synthetic-pan-marker",
        "synthetic-cvv-marker",
        "Synthetic Cardholder",
        "must not persist",
    ):
        assert forbidden not in stored


def test_signature_is_checked_before_parsing_and_exact_bytes_are_required(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
) -> None:
    compact = _raw(synthetic_event)
    differently_encoded = json.dumps(synthetic_event, indent=2).encode("utf-8")

    with TestClient(create_app(integration_settings)) as client:
        invalid = client.post(
            "/api/v1/webhooks/razorpay",
            content=differently_encoded,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": _signature(compact),
                "X-Razorpay-Event-Id": "event_changed_bytes",
            },
        )
        malformed = b"{not-json"
        signed_malformed = client.post(
            "/api/v1/webhooks/razorpay",
            content=malformed,
            headers=_headers(malformed, "event_malformed"),
        )

    assert invalid.status_code == 401
    assert invalid.json()["error"]["code"] == "invalid_signature"
    assert invalid.json()["error"]["request_id"]
    assert signed_malformed.status_code == 400
    assert signed_malformed.json()["error"]["code"] == "invalid_webhook_envelope"
    assert (
        asyncio.run(
            _events_for(
                integration_settings.database_url.get_secret_value(),
                ["event_changed_bytes", "event_malformed"],
            )
        )
        == []
    )


def test_header_content_type_and_size_errors_are_safe(
    integration_settings: Settings,
    synthetic_raw_body: bytes,
) -> None:
    with TestClient(create_app(integration_settings)) as client:
        wrong_type = client.post(
            "/api/v1/webhooks/razorpay",
            content=synthetic_raw_body,
            headers={**_headers(synthetic_raw_body), "Content-Type": "text/plain"},
        )
        missing_signature = client.post(
            "/api/v1/webhooks/razorpay",
            content=synthetic_raw_body,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Event-Id": "event_missing_signature",
            },
        )
        oversized_body = b"x" * 4_097
        oversized = client.post(
            "/api/v1/webhooks/razorpay",
            content=oversized_body,
            headers=_headers(oversized_body, "event_oversized"),
        )

    assert wrong_type.status_code == 415
    assert missing_signature.status_code == 401
    assert oversized.status_code == 413


def test_invalid_or_missing_event_id_creates_no_rows(
    integration_settings: Settings,
    synthetic_raw_body: bytes,
) -> None:
    with TestClient(create_app(integration_settings)) as client:
        invalid = client.post(
            "/api/v1/webhooks/razorpay",
            content=synthetic_raw_body,
            headers=_headers(synthetic_raw_body, "x" * 256),
        )
        missing_headers = _headers(synthetic_raw_body)
        del missing_headers["X-Razorpay-Event-Id"]
        missing = client.post(
            "/api/v1/webhooks/razorpay",
            content=synthetic_raw_body,
            headers=missing_headers,
        )

    assert invalid.status_code == 400
    assert missing.status_code == 400


def test_unsupported_signed_event_is_audited_without_observation(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
) -> None:
    event = copy.deepcopy(synthetic_event)
    event["event"] = "order.paid"
    event_id = "event_unsupported"
    raw_body = _raw(event)

    with TestClient(create_app(integration_settings)) as client:
        response = client.post(
            "/api/v1/webhooks/razorpay", content=raw_body, headers=_headers(raw_body, event_id)
        )

    assert response.status_code == 200
    rows = asyncio.run(
        _events_for(integration_settings.database_url.get_secret_value(), [event_id])
    )
    assert len(rows) == 1
    assert rows[0].processing_result == "ignored"
    observations = asyncio.run(
        _observations_for(integration_settings.database_url.get_secret_value(), "pay_synthetic")
    )
    assert all(row.webhook_event_id != rows[0].id for row in observations)


def test_out_of_order_events_remain_distinct_immutable_facts(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
) -> None:
    captured = copy.deepcopy(synthetic_event)
    captured["event"] = "payment.captured"
    captured["created_at"] = 1_700_000_200
    captured["payload"]["payment"]["entity"].update(
        {"id": "pay_out_of_order", "status": "captured", "captured": True}
    )
    authorized = copy.deepcopy(synthetic_event)
    authorized["event"] = "payment.authorized"
    authorized["created_at"] = 1_700_000_100
    authorized["payload"]["payment"]["entity"].update(
        {"id": "pay_out_of_order", "status": "authorized", "captured": False}
    )

    with TestClient(create_app(integration_settings)) as client:
        for event, event_id in (
            (captured, "event_captured_first"),
            (authorized, "event_authorized_late"),
        ):
            raw_body = _raw(event)
            assert (
                client.post(
                    "/api/v1/webhooks/razorpay",
                    content=raw_body,
                    headers=_headers(raw_body, event_id),
                ).status_code
                == 200
            )

    observations = asyncio.run(
        _observations_for(integration_settings.database_url.get_secret_value(), "pay_out_of_order")
    )
    assert [row.event_name for row in observations] == [
        "payment.authorized",
        "payment.captured",
    ]


def test_concurrent_duplicates_have_one_business_effect(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
) -> None:
    event = copy.deepcopy(synthetic_event)
    event["payload"]["payment"]["entity"]["id"] = "pay_concurrent"
    envelope = WebhookEnvelope.model_validate(event)
    raw_body = _raw(event)
    database = Database(integration_settings)

    async def ingest_once() -> bool:
        async with database.session_factory() as session:
            result = await ingest_webhook(
                session,
                provider_event_id="event_concurrent",
                envelope=envelope,
                raw_body_sha256=hashlib.sha256(raw_body).hexdigest(),
            )
            return result.duplicate

    async def run_concurrently() -> list[bool]:
        try:
            return list(await asyncio.gather(*(ingest_once() for _ in range(4))))
        finally:
            await database.close()

    duplicate_results = asyncio.run(run_concurrently())
    assert duplicate_results.count(False) == 1
    assert duplicate_results.count(True) == 3
    observations = asyncio.run(
        _observations_for(integration_settings.database_url.get_secret_value(), "pay_concurrent")
    )
    assert len(observations) == 1


async def _set_failure_trigger(database_url: str, enabled: bool) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            if enabled:
                await connection.execute(
                    text(
                        """
                        CREATE FUNCTION fail_synthetic_observation() RETURNS trigger AS $$
                        BEGIN
                            RAISE EXCEPTION 'synthetic observation failure';
                        END;
                        $$ LANGUAGE plpgsql
                        """
                    )
                )
                await connection.execute(
                    text(
                        """
                        CREATE TRIGGER fail_synthetic_observation_trigger
                        BEFORE INSERT ON payment_observations
                        FOR EACH ROW EXECUTE FUNCTION fail_synthetic_observation()
                        """
                    )
                )
            else:
                await connection.execute(
                    text("DROP TRIGGER fail_synthetic_observation_trigger ON payment_observations")
                )
                await connection.execute(text("DROP FUNCTION fail_synthetic_observation()"))
    finally:
        await engine.dispose()


def test_processing_failure_rolls_back_and_replay_succeeds(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    event = copy.deepcopy(synthetic_event)
    event["payload"]["payment"]["entity"]["id"] = "pay_replay"
    raw_body = _raw(event)
    event_id = "event_replay"
    database_url = integration_settings.database_url.get_secret_value()

    asyncio.run(_set_failure_trigger(database_url, True))
    try:
        with TestClient(create_app(integration_settings)) as client:
            failed = client.post(
                "/api/v1/webhooks/razorpay",
                content=raw_body,
                headers=_headers(raw_body, event_id),
            )
        assert failed.status_code == 503
        assert asyncio.run(_events_for(database_url, [event_id])) == []
        captured_logs = capsys.readouterr().out
        assert "person@example.invalid" not in captured_logs
        assert "pay_replay" not in captured_logs
        assert event_id not in captured_logs
    finally:
        asyncio.run(_set_failure_trigger(database_url, False))

    with TestClient(create_app(integration_settings)) as client:
        replay = client.post(
            "/api/v1/webhooks/razorpay",
            content=raw_body,
            headers=_headers(raw_body, event_id),
        )

    assert replay.status_code == 200
    assert replay.json() == {"status": "accepted", "duplicate": False}


def test_invalid_supported_payment_payload_is_not_audited(
    integration_settings: Settings,
    synthetic_event: dict[str, Any],
) -> None:
    event = copy.deepcopy(synthetic_event)
    del event["payload"]["payment"]["entity"]["amount"]
    raw_body = _raw(event)
    event_id = "event_invalid_payment"

    with TestClient(create_app(integration_settings)) as client:
        response = client.post(
            "/api/v1/webhooks/razorpay",
            content=raw_body,
            headers=_headers(raw_body, event_id),
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_payment_payload"
    assert (
        asyncio.run(_events_for(integration_settings.database_url.get_secret_value(), [event_id]))
        == []
    )


async def _count_all_rows(database_url: str) -> tuple[int, int]:
    engine = create_async_engine(database_url)
    try:
        async with AsyncSession(engine) as session:
            events = await session.scalar(select(func.count()).select_from(WebhookEvent))
            observations = await session.scalar(
                select(func.count()).select_from(PaymentObservation)
            )
            return int(events or 0), int(observations or 0)
    finally:
        await engine.dispose()
