"""Controlled failure injection against a running application.

Each scenario here breaks something on purpose and asserts the system degraded safely. None of
these behaviours is implemented by this file: every one already existed before Day 9, and the
point is to make the degradation observable rather than merely tested in the abstract.

The DB-outage scenarios use the ``dependency_overrides`` pattern established in
``tests/integration/test_health.py`` -- the closest in-process equivalent of pulling the plug.
"""

import asyncio
import copy
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import riskloom.main as main_module
from riskloom.api.dependencies import get_session
from riskloom.core.config import Settings
from riskloom.db.models import ReviewItem
from riskloom.db.models import RiskDecision as RiskDecisionRow
from riskloom.features.config import load_feature_config
from riskloom.integrations.razorpay.schemas import RazorpayOrder
from riskloom.main import create_app
from riskloom.serving.model_host import ServingBundle, ServingBundleError, load_serving_bundle
from tests.integration.test_checkout_preflight import (
    SHARED_DEVICE,
    SHARED_NETWORK,
    _locked_model,
    _request,
)

pytestmark = pytest.mark.integration

SYNTHETIC_WEBHOOK_SECRET = "synthetic_webhook_secret_for_tests_only"


def _bundle() -> ServingBundle:
    return ServingBundle(
        model=_locked_model(),
        feature_config=load_feature_config(Path("configs/features/default.json")),
        feature_dataset_id="e" * 64,
    )


class RecordingOrders:
    """Succeeds, and remembers. Used to prove an order really was created before storage died."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create_order(self, amount: int, currency: str, receipt: str) -> RazorpayOrder:
        self.calls.append({"amount": amount, "currency": currency, "receipt": receipt})
        return RazorpayOrder(
            id=f"order_{len(self.calls):026d}",
            entity="order",
            amount=amount,
            amount_paid=0,
            amount_due=amount,
            currency=currency,
            receipt=receipt,
            status="created",
            attempts=0,
            created_at=1_700_000_000,
        )

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def clean_ledger(migrated_database_url: str) -> None:
    async def truncate() -> None:
        engine = create_async_engine(migrated_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "TRUNCATE risk_decision_explanations, review_items, risk_decisions "
                        "RESTART IDENTITY CASCADE"
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(truncate())


# ------------------------------------------------------------------ webhook replay


def _raw(event: dict[str, Any]) -> bytes:
    return json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _headers(raw: bytes, event_id: str) -> dict[str, str]:
    signature = hmac.new(SYNTHETIC_WEBHOOK_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
        "X-Razorpay-Event-Id": event_id,
    }


async def _observation_count(url: str, payment_id: str) -> int:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            return int(
                (
                    await connection.execute(
                        text(
                            "SELECT count(*) FROM payment_observations "
                            "WHERE provider_payment_id = :pid"
                        ),
                        {"pid": payment_id},
                    )
                ).scalar()
                or 0
            )
    finally:
        await engine.dispose()


def test_injecting_a_duplicate_webhook_creates_no_second_business_effect(
    integration_settings: Settings, synthetic_event: dict[str, Any]
) -> None:
    """The Day 1 guarantee, exercised as an injected fault rather than a happy-path test."""

    event = copy.deepcopy(synthetic_event)
    event["payload"]["payment"]["entity"]["id"] = "pay_inject_duplicate"
    raw = _raw(event)
    headers = _headers(raw, "event_inject_duplicate")

    with TestClient(create_app(integration_settings)) as client:
        first = client.post("/api/v1/webhooks/razorpay", content=raw, headers=headers)
        for _ in range(4):
            repeat = client.post("/api/v1/webhooks/razorpay", content=raw, headers=headers)
            assert repeat.status_code == 200
            assert repeat.json()["duplicate"] is True

    assert first.status_code == 200
    assert first.json()["duplicate"] is False
    observations = asyncio.run(
        _observation_count(
            integration_settings.database_url.get_secret_value(), "pay_inject_duplicate"
        )
    )
    assert observations == 1, "five deliveries, one business effect"


def test_injecting_an_out_of_order_webhook_preserves_both_facts(
    integration_settings: Settings, synthetic_event: dict[str, Any]
) -> None:
    """A late earlier event must not be discarded, and must not overwrite the later one."""

    captured = copy.deepcopy(synthetic_event)
    captured["event"] = "payment.captured"
    captured["created_at"] = 1_700_000_900
    captured["payload"]["payment"]["entity"].update(
        {"id": "pay_inject_ooo", "status": "captured", "captured": True}
    )
    authorized = copy.deepcopy(synthetic_event)
    authorized["event"] = "payment.authorized"
    authorized["created_at"] = 1_700_000_100
    authorized["payload"]["payment"]["entity"].update(
        {"id": "pay_inject_ooo", "status": "authorized", "captured": False}
    )

    with TestClient(create_app(integration_settings)) as client:
        for event, event_id in (
            (captured, "event_inject_ooo_late"),
            (authorized, "event_inject_ooo_early"),
        ):
            raw = _raw(event)
            response = client.post(
                "/api/v1/webhooks/razorpay", content=raw, headers=_headers(raw, event_id)
            )
            assert response.status_code == 200
            assert response.json()["duplicate"] is False

    assert (
        asyncio.run(
            _observation_count(
                integration_settings.database_url.get_secret_value(), "pay_inject_ooo"
            )
        )
        == 2
    ), "both events survive as distinct append-only facts"


# ------------------------------------------------------------------ storage outage


class DeadSession:
    """A session whose every statement fails the way a severed connection does."""

    def __init__(self, fail_after: int = 0) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.calls > self.fail_after:
            raise OperationalError("SELECT 1", {}, Exception("connection closed"))

    async def scalar(self, *_a: Any, **_k: Any) -> Any:
        self._maybe_fail()
        return None

    async def execute(self, *_a: Any, **_k: Any) -> Any:
        self._maybe_fail()
        return None

    async def get(self, *_a: Any, **_k: Any) -> Any:
        self._maybe_fail()
        return None

    def begin(self) -> Any:
        outer = self

        class _Transaction:
            async def __aenter__(self) -> None:
                outer._maybe_fail()

            async def __aexit__(self, *_a: Any) -> None:
                return None

        return _Transaction()


def test_preflight_refuses_when_storage_is_unavailable(
    integration_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: a 503, never a 200 ALLOW the ledger cannot back."""

    monkeypatch.setattr(main_module, "load_serving_bundle", lambda **_: _bundle())
    monkeypatch.setattr(main_module, "RazorpayOrdersClient", lambda *_a, **_k: RecordingOrders())
    app = create_app(integration_settings)

    async def dead_session() -> Any:
        yield DeadSession()

    app.dependency_overrides[get_session] = dead_session

    with TestClient(app) as client:
        response = client.post("/api/v1/checkout/preflight", json=_request(1))

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"
    assert "OperationalError" not in response.text
    assert "connection closed" not in response.text


def test_a_storage_outage_leaves_no_partial_ledger_state(
    integration_settings: Settings, migrated_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "load_serving_bundle", lambda **_: _bundle())
    monkeypatch.setattr(main_module, "RazorpayOrdersClient", lambda *_a, **_k: RecordingOrders())
    app = create_app(integration_settings)

    async def dead_session() -> Any:
        yield DeadSession()

    app.dependency_overrides[get_session] = dead_session

    with TestClient(app) as client:
        assert client.post("/api/v1/checkout/preflight", json=_request(2)).status_code == 503

    async def counts() -> tuple[int, int]:
        engine = create_async_engine(migrated_database_url)
        try:
            async with AsyncSession(engine) as session:
                decisions = int(
                    await session.scalar(select(func.count()).select_from(RiskDecisionRow)) or 0
                )
                reviews = int(
                    await session.scalar(select(func.count()).select_from(ReviewItem)) or 0
                )
            return decisions, reviews
        finally:
            await engine.dispose()

    assert asyncio.run(counts()) == (0, 0)


class FailsOnFinalWrite:
    """A real session that dies only when the ledger is finalised.

    Wrapping a genuine session matters here: the claim has to actually commit and the order has to
    actually be created, or the test would not be exercising the window it claims to. Only the
    second ``begin()`` -- the pending-to-final transition -- is severed.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.transactions = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def begin(self) -> Any:
        self.transactions += 1
        if self.transactions < 2:
            return self._session.begin()

        class _Severed:
            async def __aenter__(self) -> None:
                raise OperationalError("UPDATE risk_decisions", {}, Exception("connection closed"))

            async def __aexit__(self, *_a: Any) -> None:
                return None

        return _Severed()


def test_a_ledger_write_failure_after_an_order_is_logged_reconcilably(
    integration_settings: Settings,
    migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one window with an external consequence.

    The claim commits, an order is genuinely created upstream, and only then does storage die. The
    caller still gets 503 rather than an unbacked ALLOW, and the order id is logged under a
    distinct identity so the orphan is reconcilable rather than silent.
    """

    orders = RecordingOrders()
    monkeypatch.setattr(main_module, "load_serving_bundle", lambda **_: _bundle())
    monkeypatch.setattr(main_module, "RazorpayOrdersClient", lambda *_a, **_k: orders)

    app = create_app(integration_settings)
    engine = create_async_engine(migrated_database_url)

    async def wrapped_session() -> Any:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield FailsOnFinalWrite(session)

    app.dependency_overrides[get_session] = wrapped_session

    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/checkout/preflight", json=_request(4))
    finally:
        asyncio.run(engine.dispose())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"

    # The order really was created, which is exactly why the orphan needs to be findable.
    assert len(orders.calls) == 1
    logged = capsys.readouterr().out
    assert "preflight_ledger_write_failed" in logged
    assert "order_00000000000000000000000001" in logged, "the order id must be reconcilable"

    # The claimed row survives as `pending`; it is never left half-final.
    async def status_of() -> str | None:
        probe = create_async_engine(migrated_database_url)
        try:
            async with AsyncSession(probe) as session:
                return await session.scalar(
                    select(RiskDecisionRow.status).where(
                        RiskDecisionRow.event_id == _request(4)["event_id"]
                    )
                )
        finally:
            await probe.dispose()

    assert asyncio.run(status_of()) == "pending"


# ------------------------------------------------------------------ startup binding


def test_a_missing_model_file_refuses_to_start(tmp_path: Path) -> None:
    """Fail closed at startup: no model, no service."""

    with pytest.raises(ServingBundleError) as caught:
        load_serving_bundle(
            feature_config_path=Path("configs/features/default.json"),
            modeling_config_path=Path("configs/modeling/default.json"),
            model_directory=tmp_path / "absent",
            feature_manifest_path=tmp_path / "absent" / "manifest.json",
        )
    assert str(caught.value).startswith("serving_")


def test_a_corrupted_model_file_refuses_to_start(tmp_path: Path) -> None:
    """A truncated artifact must be rejected, not partially loaded."""

    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model.json").write_text('{"model_id": "truncated"', encoding="utf-8")
    (directory / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ServingBundleError) as caught:
        load_serving_bundle(
            feature_config_path=Path("configs/features/default.json"),
            modeling_config_path=Path("configs/modeling/default.json"),
            model_directory=directory,
            feature_manifest_path=directory / "manifest.json",
        )
    assert str(caught.value).startswith("serving_")


def test_the_startup_error_never_leaks_a_path_or_a_secret(tmp_path: Path) -> None:
    with pytest.raises(ServingBundleError) as caught:
        load_serving_bundle(
            feature_config_path=Path("configs/features/default.json"),
            modeling_config_path=Path("configs/modeling/default.json"),
            model_directory=tmp_path / "absent",
            feature_manifest_path=tmp_path / "absent" / "manifest.json",
        )
    message = str(caught.value)
    assert str(tmp_path) not in message
    assert message.count(" ") == 0, "error identities are short stable tokens, not prose"


# ------------------------------------------------------------------ shared-token guard


def test_injection_helpers_use_only_synthetic_tokens() -> None:
    """Nothing in this suite may carry a real-looking identifier."""

    for token in (SHARED_DEVICE, SHARED_NETWORK):
        assert token.split("_", 1)[1] in {"a" * 32, "b" * 32}
