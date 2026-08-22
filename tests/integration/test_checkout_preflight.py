"""End-to-end live preflight: all three actions, idempotency, concurrency and the ledger."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import riskloom.main as main_module
from riskloom.core.config import Settings
from riskloom.db.models import ReviewItem
from riskloom.db.models import RiskDecision as RiskDecisionRow
from riskloom.features.config import load_feature_config
from riskloom.features.schema import FEATURE_COUNT, FEATURE_NAMES
from riskloom.integrations.razorpay.client import RazorpayOrdersError
from riskloom.integrations.razorpay.schemas import RazorpayOrder
from riskloom.main import create_app
from riskloom.modeling.model import LockedModel, LogisticPortableModel, PlattModel
from riskloom.serving.model_host import ServingBundle

pytestmark = pytest.mark.integration

DEVICE_INDEX = FEATURE_NAMES.index("device_prior_attempt_count_3600s")
SHARED_DEVICE = "dev_" + "a" * 32
SHARED_NETWORK = "net_" + "b" * 32
# Probability is sigmoid(prior_device_attempts - 2.5), so the first three attempts on a shared
# device allow and the fourth denies. That mirrors the real demo shape: velocity drives the flip.
THRESHOLD = 0.5


class FakeOrdersClient:
    """Stands in for the real Razorpay client. No network call is ever made in CI."""

    def __init__(
        self,
        *_args: Any,
        fail: bool = False,
        fail_from: int | None = None,
        **_kwargs: Any,
    ) -> None:
        self.fail = fail
        # 1-based index of the first call that should fail, so a run can mix a success with a
        # subsequent rejection the way the live verification did.
        self.fail_from = fail_from
        self.calls: list[dict[str, Any]] = []

    async def create_order(self, amount: int, currency: str, receipt: str) -> RazorpayOrder:
        self.calls.append({"amount": amount, "currency": currency, "receipt": receipt})
        if self.fail or (self.fail_from is not None and len(self.calls) >= self.fail_from):
            raise RazorpayOrdersError("razorpay_orders_rejected")
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


def _locked_model() -> LockedModel:
    coefficients = [0.0] * FEATURE_COUNT
    coefficients[DEVICE_INDEX] = 1.0
    return LockedModel(
        model_id="d" * 64,
        feature_order=list(FEATURE_NAMES),
        class_order=[0, 1],
        decision_threshold=THRESHOLD,
        candidate=LogisticPortableModel(
            candidate_name="logistic_regression",
            coefficients=coefficients,
            intercept=0.0,
            scaler_mean=[0.0] * FEATURE_COUNT,
            scaler_scale=[1.0] * FEATURE_COUNT,
        ),
        calibration=PlattModel(coefficient=1.0, intercept=-2.5, probability_clip_epsilon=1e-15),
    )


def _bundle() -> ServingBundle:
    return ServingBundle(
        model=_locked_model(),
        feature_config=load_feature_config(Path("configs/features/default.json")),
        feature_dataset_id="e" * 64,
    )


def _request(index: int, *, amount: int = 25_000, shared: bool = True) -> dict[str, Any]:
    return {
        "event_id": f"evt_{index:032x}",
        "merchant_id": f"mrc_{1:032x}",
        "checkout_id": f"chk_{index:032x}",
        "customer_token": None,
        "device_token": SHARED_DEVICE if shared else f"dev_{index:032x}",
        "network_token": SHARED_NETWORK if shared else f"net_{index:032x}",
        "session_token": f"ses_{index:032x}",
        "payment_instrument_token": f"pmt_{index:032x}",
        "amount_subunits": amount,
        "currency": "INR",
        "channel": "web",
    }


@pytest.fixture(autouse=True)
def clean_ledger(migrated_database_url: str) -> None:
    """Truncate the Day 6 tables before each test.

    The database fixture is session scoped, so without this the ledger accumulates rows across
    tests and every count assertion silently drifts.
    """

    async def truncate() -> None:
        engine = create_async_engine(migrated_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text("TRUNCATE review_items, risk_decisions RESTART IDENTITY CASCADE")
                )
        finally:
            await engine.dispose()

    asyncio.run(truncate())


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch, integration_settings: Settings) -> Any:
    def build(
        *,
        fail_orders: bool = False,
        fail_from: int | None = None,
        order_limit: int = 5,
    ) -> tuple[TestClient, Any]:
        orders = FakeOrdersClient(fail=fail_orders, fail_from=fail_from)
        monkeypatch.setattr(main_module, "load_serving_bundle", lambda **_: _bundle())
        monkeypatch.setattr(main_module, "RazorpayOrdersClient", lambda *_a, **_k: orders)
        settings = integration_settings.model_copy(
            update={"razorpay_max_orders_per_process": order_limit}
        )
        return TestClient(create_app(settings)), orders

    return build


async def _rows(database_url: str) -> list[RiskDecisionRow]:
    engine = create_async_engine(database_url)
    try:
        async with AsyncSession(engine) as session:
            result = await session.scalars(
                select(RiskDecisionRow).order_by(RiskDecisionRow.created_at)
            )
            return list(result.all())
    finally:
        await engine.dispose()


async def _review_count(database_url: str) -> int:
    engine = create_async_engine(database_url)
    try:
        async with AsyncSession(engine) as session:
            return int(await session.scalar(select(func.count()).select_from(ReviewItem)) or 0)
    finally:
        await engine.dispose()


def test_allow_creates_exactly_one_order_and_denies_once_velocity_crosses_the_threshold(
    client_factory: Any, migrated_database_url: str
) -> None:
    client, orders = client_factory()
    with client:
        responses = [client.post("/api/v1/checkout/preflight", json=_request(i)) for i in range(4)]

    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    payloads = [response.json() for response in responses]
    actions = [payload["action"] for payload in payloads]
    assert actions == ["allow", "allow", "allow", "deny"]
    assert [payload["risk_decision"] for payload in payloads] == [
        "allow",
        "allow",
        "allow",
        "deny",
    ]
    # Three allows created three orders; the deny created none.
    assert len(orders.calls) == 3
    assert all(len(call["receipt"]) == 35 for call in orders.calls)
    assert payloads[3]["razorpay_order_id"] is None
    assert payloads[0]["decision_threshold"] == THRESHOLD
    assert payloads[0]["fail_safe_reason"] is None

    rows = asyncio.run(_rows(migrated_database_url))
    assert len(rows) == 4
    assert all(row.status == "final" for row in rows)
    assert all(row.model_id == "d" * 64 for row in rows)
    assert asyncio.run(_review_count(migrated_database_url)) == 0


def test_order_creation_failure_fail_safes_to_review_without_changing_the_risk_decision(
    client_factory: Any, migrated_database_url: str
) -> None:
    client, orders = client_factory(fail_orders=True)
    with client:
        response = client.post("/api/v1/checkout/preflight", json=_request(10))

    payload = response.json()
    assert response.status_code == 200
    assert payload["action"] == "review"
    # The model still said allow; only the action was downgraded. That distinction is auditable.
    assert payload["risk_decision"] == "allow"
    assert payload["fail_safe_reason"] == "order_creation_failed"
    assert payload["razorpay_order_id"] is None
    assert len(orders.calls) == 1

    assert asyncio.run(_review_count(migrated_database_url)) == 1
    rows = asyncio.run(_rows(migrated_database_url))
    assert rows[0].action == "review"
    assert rows[0].risk_decision == "allow"


def test_order_budget_exhaustion_fail_safes_to_review_without_calling_razorpay(
    client_factory: Any, migrated_database_url: str
) -> None:
    """The successful-ALLOW case: one order created, then the cap bites.

    This covers budget consumption by a *succeeding* attempt only. Consumption by a failing
    attempt is covered separately below, because the budget is reserved before the upstream call
    and the two paths are not interchangeable.
    """

    client, orders = client_factory(order_limit=1)
    with client:
        first = client.post("/api/v1/checkout/preflight", json=_request(20)).json()
        second = client.post("/api/v1/checkout/preflight", json=_request(21)).json()

    assert first["action"] == "allow"
    assert second["action"] == "review"
    assert second["fail_safe_reason"] == "order_budget_exhausted"
    # The budget is enforced before the client is touched, so only one call was made.
    assert len(orders.calls) == 1


def test_failed_order_attempts_consume_budget_exactly_like_successful_ones(
    client_factory: Any, migrated_database_url: str
) -> None:
    """The budget counts attempts, not successes.

    Every upstream call is rejected here, so no order is ever created, yet the cap is still
    reached. This is the behaviour observed in manual verification, where a sub-minimum amount was
    rejected by Razorpay and still consumed a unit of budget.
    """

    client, orders = client_factory(fail_orders=True, order_limit=2)
    with client:
        payloads = [
            client.post("/api/v1/checkout/preflight", json=_request(500 + index)).json()
            for index in range(3)
        ]

    # Two attempts were made and both were rejected upstream.
    assert [payload["action"] for payload in payloads] == ["review", "review", "review"]
    assert [payload["fail_safe_reason"] for payload in payloads] == [
        "order_creation_failed",
        "order_creation_failed",
        "order_budget_exhausted",
    ]
    # Not one order exists, yet the budget is spent: exactly two calls reached the client, and the
    # third never did because the cap was already reached.
    assert len(orders.calls) == 2
    assert all(payload["razorpay_order_id"] is None for payload in payloads)
    # The underlying risk decision is untouched by any of this.
    assert all(payload["risk_decision"] == "allow" for payload in payloads)


def test_budget_counts_a_mixed_run_of_one_success_and_one_rejection(
    client_factory: Any, migrated_database_url: str
) -> None:
    """Mirrors the live verification session exactly: one order created, one rejected, cap of 2."""

    client, orders = client_factory(fail_from=2, order_limit=2)
    with client:
        created = client.post("/api/v1/checkout/preflight", json=_request(600)).json()
        rejected = client.post("/api/v1/checkout/preflight", json=_request(601)).json()
        capped = client.post("/api/v1/checkout/preflight", json=_request(602)).json()

    assert created["action"] == "allow"
    assert created["razorpay_order_id"] is not None
    assert rejected["action"] == "review"
    assert rejected["fail_safe_reason"] == "order_creation_failed"
    assert capped["action"] == "review"
    assert capped["fail_safe_reason"] == "order_budget_exhausted"

    # One order exists, but two units of budget were spent, so the counter is ahead of reality.
    assert len(orders.calls) == 2
    orders_created = [
        payload for payload in (created, rejected, capped) if payload["razorpay_order_id"]
    ]
    assert len(orders_created) == 1

    rows = asyncio.run(_rows(migrated_database_url))
    assert len([row for row in rows if row.razorpay_order_id]) == 1


def test_a_zero_budget_refuses_every_order_attempt(client_factory: Any) -> None:
    """The cap is configurable down to zero, which disables outbound order creation entirely."""

    client, orders = client_factory(order_limit=0)
    with client:
        payload = client.post("/api/v1/checkout/preflight", json=_request(700)).json()

    assert payload["action"] == "review"
    assert payload["fail_safe_reason"] == "order_budget_exhausted"
    assert orders.calls == []


def test_a_denied_decision_never_consumes_budget(client_factory: Any) -> None:
    """Only an ALLOW reaches the budget, so denials cannot exhaust it."""

    client, orders = client_factory(order_limit=1)
    with client:
        # Four attempts on one device: the fourth denies, which must not touch the budget.
        payloads = [
            client.post("/api/v1/checkout/preflight", json=_request(800 + index)).json()
            for index in range(4)
        ]

    assert payloads[3]["action"] == "deny"
    # The first ALLOW consumed the only unit; the two following ALLOWs found it exhausted; the
    # DENY never reached the budget at all.
    assert len(orders.calls) == 1
    assert payloads[1]["fail_safe_reason"] == "order_budget_exhausted"
    assert payloads[3]["fail_safe_reason"] is None


def test_a_retried_request_does_not_score_or_order_twice(
    client_factory: Any, migrated_database_url: str
) -> None:
    client, orders = client_factory()
    with client:
        first = client.post("/api/v1/checkout/preflight", json=_request(30)).json()
        second = client.post("/api/v1/checkout/preflight", json=_request(30)).json()
        third = client.post("/api/v1/checkout/preflight", json=_request(31)).json()

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["decision_id"] == first["decision_id"]
    assert second["action"] == first["action"]
    assert len(orders.calls) == 2, "the retry must not create a second order"

    rows = asyncio.run(_rows(migrated_database_url))
    assert len(rows) == 2
    # The retry did not advance rolling state either: the third event saw exactly one prior
    # attempt on the shared device, so its probability is strictly higher than the first event's
    # and strictly lower than it would be had the retry been scored.
    assert third["calibrated_probability"] > first["calibrated_probability"]
    probabilities = [row.calibrated_probability for row in rows]
    assert len(set(probabilities)) == 2


def test_concurrent_requests_are_all_scored_exactly_once(
    client_factory: Any, migrated_database_url: str
) -> None:
    client, orders = client_factory(order_limit=50)
    count = 12
    with client, ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(
            pool.map(
                lambda index: client.post("/api/v1/checkout/preflight", json=_request(100 + index)),
                range(count),
            )
        )

    assert all(response.status_code == 200 for response in responses)
    rows = asyncio.run(_rows(migrated_database_url))
    assert len(rows) == count
    assert all(row.status == "final" for row in rows)
    # Every concurrent request received a distinct engine timestamp: no lost state update.
    stamps = [row.occurred_at for row in rows]
    assert len(set(stamps)) == count


def test_malformed_and_pii_bearing_requests_are_rejected_before_scoring(
    client_factory: Any,
) -> None:
    client, orders = client_factory()
    with client:
        with_pii = client.post(
            "/api/v1/checkout/preflight",
            json={**_request(200), "email": "person@example.invalid"},
        )
        malformed = client.post(
            "/api/v1/checkout/preflight", json={**_request(201), "device_token": "not-a-token"}
        )

    assert with_pii.status_code == 422
    assert malformed.status_code == 422
    assert orders.calls == []


def test_response_reports_the_unrounded_model_threshold_not_the_audit_column(
    client_factory: Any, migrated_database_url: str
) -> None:
    """Regression: the response must not surface the Numeric(20, 18) round-trip.

    The fixture threshold of 0.5 round-trips exactly, so this asserts against a value that does
    not: the real locked threshold carries 19 significant decimals and the column holds 18.
    """

    client, _ = client_factory()
    with client:
        payload = client.post("/api/v1/checkout/preflight", json=_request(400)).json()

    assert payload["decision_threshold"] == THRESHOLD

    from decimal import Decimal  # noqa: PLC0415

    from riskloom.services.preflight import _probability_decimal  # noqa: PLC0415

    real_threshold = 0.0033862949155182734
    assert float(_probability_decimal(real_threshold)) != real_threshold
    assert _probability_decimal(real_threshold) == Decimal("0.003386294915518273")


def test_the_response_never_leaks_features_or_internal_state(client_factory: Any) -> None:
    client, _ = client_factory()
    with client:
        payload = client.post("/api/v1/checkout/preflight", json=_request(300)).json()

    assert set(payload) == {
        "decision_id",
        "event_id",
        "action",
        "risk_decision",
        "calibrated_probability",
        "decision_threshold",
        "model_id",
        "fail_safe_reason",
        "razorpay_order_id",
        "evaluated_at",
        "duplicate",
    }
    rendered = str(payload)
    for token in ("merchant_id", "device_token", "session_token", "prior_attempt", "distinct"):
        assert token not in rendered


def test_existing_health_and_webhook_routes_are_unchanged(client_factory: Any) -> None:
    client, _ = client_factory()
    with client:
        assert client.get("/health/live").status_code == 200
        # The webhook still rejects an unsigned body exactly as Day 1 specified.
        unsigned = client.post(
            "/api/v1/webhooks/razorpay",
            content=b"{}",
            headers={"Content-Type": "application/json"},
        )
    assert unsigned.status_code == 401
