"""The dashboard's demo burst, exercised against the real scoring endpoint.

The burst is client-side orchestration of `POST /api/v1/checkout/preflight` and implements no
decision logic of its own, so what needs proving is the *shape* it sends and the bounds it inherits:
four correlated attempts, tokens in the exact form the browser produces, an escalating verdict
sequence, and no more Razorpay order attempts than the process cap allows.

The request bodies here are built the same way `static/app.js` builds them, including the
`crypto.randomUUID().replaceAll("-", "")` transformation, so a change to that shape fails here
rather than in front of a judge.
"""

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import riskloom.main as main_module
from riskloom.core.config import Settings
from riskloom.features.config import load_feature_config
from riskloom.main import create_app
from riskloom.serving.model_host import ServingBundle
from tests.integration.test_checkout_preflight import FakeOrdersClient, _locked_model

pytestmark = pytest.mark.integration

# Mirrors the constants in static/app.js. Fixed rather than random, so a second burst finds the
# device already hot and denies immediately without attempting another order.
BURST_SIZE = 4
DEMO_DEVICE = "dev_deadbeefdeadbeefdeadbeefdeadbeef"
DEMO_NETWORK = "net_cafebabecafebabecafebabecafebabe"
DEMO_MERCHANT = "mrc_decafbaddecafbaddecafbaddecafbad"
TOKEN_PATTERN = re.compile(r"^(evt|chk|ses|pmt)_[0-9a-f]{32}$")

STATIC_APP = Path("static/app.js")


def demo_token(prefix: str) -> str:
    """Exactly what `crypto.randomUUID().replaceAll("-", "")` yields in the browser.

    `crypto.randomUUID()` returns 36 characters in 8-4-4-4-12 form: 32 lowercase hex digits plus
    four hyphens. Python's `uuid4()` has the identical string form, so stripping hyphens here
    reproduces the client's transformation rather than approximating it.
    """

    return f"{prefix}_{str(uuid.uuid4()).replace('-', '')}"


def burst_attempt() -> dict[str, Any]:
    return {
        "event_id": demo_token("evt"),
        "merchant_id": DEMO_MERCHANT,
        "checkout_id": demo_token("chk"),
        "customer_token": None,
        "device_token": DEMO_DEVICE,
        "network_token": DEMO_NETWORK,
        "session_token": demo_token("ses"),
        "payment_instrument_token": demo_token("pmt"),
        "amount_subunits": 25_000,
        "currency": "INR",
        "channel": "web",
    }


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


@pytest.fixture
def burst_client(monkeypatch: pytest.MonkeyPatch, integration_settings: Settings) -> Any:
    def build(order_limit: int = 5) -> tuple[TestClient, FakeOrdersClient]:
        orders = FakeOrdersClient()
        monkeypatch.setattr(
            main_module,
            "load_serving_bundle",
            lambda **_: ServingBundle(
                model=_locked_model(),
                feature_config=load_feature_config(Path("configs/features/default.json")),
                feature_dataset_id="e" * 64,
            ),
        )
        monkeypatch.setattr(main_module, "RazorpayOrdersClient", lambda *_a, **_k: orders)
        settings = integration_settings.model_copy(
            update={"razorpay_max_orders_per_process": order_limit}
        )
        return TestClient(create_app(settings)), orders

    return build


def fire(client: TestClient) -> list[dict[str, Any]]:
    responses = []
    for _ in range(BURST_SIZE):
        response = client.post("/api/v1/checkout/preflight", json=burst_attempt())
        assert response.status_code == 200, response.text
        responses.append(response.json())
    return responses


# --------------------------------------------------------------------------- shape and bounds


def test_a_burst_escalates_from_allow_to_deny(burst_client: Any) -> None:
    """The measured shape: velocity builds on a shared device until the threshold is crossed."""

    client, orders = burst_client()
    with client:
        decisions = fire(client)

    actions = [d["action"] for d in decisions]
    assert len(actions) == BURST_SIZE
    assert actions[0] == "allow", actions
    assert actions[-1] == "deny", actions
    assert "deny" in actions and "allow" in actions, actions

    # Only the allows attempt an order; a deny creates none. That is what bounds the demo's spend.
    assert len(orders.calls) == actions.count("allow")


def test_every_attempt_is_a_new_event_never_a_replay(burst_client: Any) -> None:
    client, _ = burst_client()
    with client:
        decisions = fire(client)

    ids = [d["event_id"] for d in decisions]
    assert len(set(ids)) == BURST_SIZE, ids
    assert not any(d["duplicate"] for d in decisions)


def test_the_burst_shares_one_device_and_one_network(burst_client: Any) -> None:
    """Correlation is the point: without it there is no hub for the graph to form."""

    client, _ = burst_client()
    with client:
        decisions = fire(client)
        # Read inside the lifespan: the app disposes its engine on exit.
        detail = [
            client.get(f"/api/v1/dashboard/decisions/{d['decision_id']}").json()["decision"]
            for d in decisions
        ]
    assert {row["device_token"] for row in detail} == {DEMO_DEVICE}
    assert {row["network_token"] for row in detail} == {DEMO_NETWORK}
    # Instruments rotate, which is what card testing actually looks like.
    assert len({row["payment_instrument_token"] for row in detail}) == BURST_SIZE


def test_a_second_burst_denies_immediately_and_costs_no_orders(burst_client: Any) -> None:
    """Why the demo device is fixed rather than random.

    After the first burst the device carries enough history that every later attempt denies at
    once, so repeat clicks are free. A fresh device per click would spend the order cap instead.
    """

    client, orders = burst_client()
    with client:
        fire(client)
        first_round_orders = len(orders.calls)
        second = fire(client)

    assert [d["action"] for d in second] == ["deny"] * BURST_SIZE
    assert len(orders.calls) == first_round_orders, "a repeat burst must attempt no new order"


def test_the_order_cap_still_bounds_the_burst(burst_client: Any) -> None:
    """The demo inherits the process cap; it does not get its own, and cannot exceed it."""

    client, orders = burst_client(order_limit=1)
    with client:
        decisions = fire(client)

    assert len(orders.calls) <= 1
    fail_safed = [d for d in decisions if d["fail_safe_reason"] == "order_budget_exhausted"]
    assert fail_safed, "past the cap an allow must fail-safe to review, not call Razorpay"
    assert all(d["action"] in {"allow", "review", "deny"} for d in decisions)


# --------------------------------------------------------------------------- token shape


def test_tokens_use_the_browser_transformation_and_validate() -> None:
    attempt = burst_attempt()
    for field in ("event_id", "checkout_id", "session_token", "payment_instrument_token"):
        assert TOKEN_PATTERN.fullmatch(attempt[field]), (field, attempt[field])


def test_the_unstripped_uuid_form_is_rejected(burst_client: Any) -> None:
    """The hyphenated 36-character form must not be accepted.

    This is the specific mistake the transformation exists to avoid, so it is asserted against the
    real endpoint rather than assumed from the regex.
    """

    client, _ = burst_client()
    body = burst_attempt()
    body["event_id"] = f"evt_{uuid.uuid4()}"  # 36 chars, hyphens intact
    with client:
        assert client.post("/api/v1/checkout/preflight", json=body).status_code == 422


# --------------------------------------------------------------------------- client parity


def test_the_client_and_this_test_agree_on_the_burst_constants() -> None:
    """A drift between app.js and this test would make the whole file test nothing."""

    source = STATIC_APP.read_text(encoding="utf-8")
    assert f"const BURST_SIZE = {BURST_SIZE};" in source
    assert f'const DEMO_DEVICE = "{DEMO_DEVICE}";' in source
    assert f'const DEMO_NETWORK = "{DEMO_NETWORK}";' in source
    assert f'const DEMO_MERCHANT = "{DEMO_MERCHANT}";' in source
    assert 'crypto.randomUUID().replaceAll("-", "")' in source


def test_the_client_labels_the_burst_as_a_demo_control() -> None:
    """It must never read as something a merchant operator would use."""

    html = Path("static/index.html").read_text(encoding="utf-8")
    assert 'class="demo-tag"' in html
    assert "demo trigger" in html
    assert "would not have this control" in html
