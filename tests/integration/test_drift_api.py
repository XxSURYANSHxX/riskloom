"""Drift endpoint: read-only, honest about small samples, and free of PII."""

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from riskloom.core.config import Settings
from riskloom.db.models import RiskDecision
from riskloom.drift.psi import PSI_MINIMUM_ROWS
from riskloom.main import create_app

pytestmark = pytest.mark.integration

DRIFT = "/api/v1/dashboard/drift"
NON_GET = ("POST", "PUT", "PATCH", "DELETE", "OPTIONS")
THRESHOLD = Decimal("0.003386294915518273")
LOCKED = Path("artifacts/evaluations/development/evaluation.json")

locked_only = pytest.mark.skipif(not LOCKED.exists(), reason="locked evaluation artifact absent")


def _row(index: int, probability: str, *, age_hours: float = 0.5) -> RiskDecision:
    now = datetime.now(UTC) - timedelta(hours=age_hours)
    return RiskDecision(
        id=uuid4(),
        event_id=f"evt_{index:032x}",
        merchant_id=f"mrc_{1:032x}",
        checkout_id=f"chk_{index:032x}",
        session_token=f"ses_{index:032x}",
        payment_instrument_token=f"pmt_{index:032x}",
        customer_token=None,
        device_token=f"dev_{index:032x}",
        network_token=f"net_{index:032x}",
        amount_subunits=25_000,
        currency="INR",
        channel="web",
        occurred_at=now,
        calibrated_probability=Decimal(probability),
        decision_threshold=THRESHOLD,
        risk_decision="deny" if Decimal(probability) >= THRESHOLD else "allow",
        action="deny" if Decimal(probability) >= THRESHOLD else "allow",
        fail_safe_reason=None,
        model_id="d" * 64,
        feature_dataset_id="e" * 64,
        feature_schema_version="1.0.0",
        feature_engine_version="1.0.0",
        razorpay_order_id=None,
        status="final",
    )


def seed(url: str, rows: list[RiskDecision]) -> None:
    async def run() -> None:
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "TRUNCATE risk_decision_explanations, review_items, risk_decisions "
                        "RESTART IDENTITY CASCADE"
                    )
                )
            async with AsyncSession(engine, expire_on_commit=False) as session, session.begin():
                for row in rows:
                    session.add(row)
        finally:
            await engine.dispose()

    asyncio.run(run())


@pytest.fixture
def client(integration_settings: Settings) -> TestClient:
    return TestClient(create_app(integration_settings))


# ------------------------------------------------------------------ read-only


@pytest.mark.parametrize("method", NON_GET)
def test_the_drift_endpoint_rejects_every_non_get_method(
    client: TestClient, migrated_database_url: str, method: str
) -> None:
    seed(migrated_database_url, [])
    with client:
        assert client.request(method, DRIFT).status_code == 405


def test_day_seven_endpoints_keep_their_get_only_matrix(
    client: TestClient, migrated_database_url: str
) -> None:
    """Adding a router must not loosen anything that already existed."""

    seed(migrated_database_url, [])
    with client:
        for path in ("summary", "decisions", "coordination", "model"):
            assert client.post(f"/api/v1/dashboard/{path}").status_code == 405


def test_reading_drift_never_writes_a_row(client: TestClient, migrated_database_url: str) -> None:
    seed(migrated_database_url, [_row(index, "0.001772338243283710") for index in range(5)])

    async def count() -> tuple[int, int]:
        engine = create_async_engine(migrated_database_url)
        try:
            async with engine.begin() as connection:
                decisions = int(
                    (await connection.execute(text("SELECT count(*) FROM risk_decisions"))).scalar()
                    or 0
                )
                explanations = int(
                    (
                        await connection.execute(
                            text("SELECT count(*) FROM risk_decision_explanations")
                        )
                    ).scalar()
                    or 0
                )
            return decisions, explanations
        finally:
            await engine.dispose()

    before = asyncio.run(count())
    with client:
        for _ in range(3):
            assert client.get(DRIFT).status_code == 200
    assert asyncio.run(count()) == before


# ------------------------------------------------------------------ small samples


@locked_only
def test_a_small_ledger_reports_insufficient_data_rather_than_a_band(
    client: TestClient, migrated_database_url: str
) -> None:
    """The honest answer for a 13-row ledger is 'not enough data', not a decimal."""

    seed(migrated_database_url, [_row(index, "0.001772338243283710") for index in range(13)])
    with client:
        payload = client.get(DRIFT).json()

    assert payload["status"] == "insufficient_data"
    assert payload["psi"] is None
    assert payload["band"] is None
    assert payload["observed_rows"] == 13
    assert payload["minimum_rows"] == PSI_MINIMUM_ROWS


@locked_only
def test_an_empty_window_is_insufficient_not_an_error(
    client: TestClient, migrated_database_url: str
) -> None:
    seed(migrated_database_url, [])
    with client:
        response = client.get(DRIFT)
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_data"
    assert response.json()["observed_rows"] == 0


@locked_only
def test_rows_outside_the_window_are_excluded(
    client: TestClient, migrated_database_url: str
) -> None:
    recent = [_row(index, "0.001772338243283710", age_hours=1) for index in range(5)]
    stale = [_row(100 + index, "0.001772338243283710", age_hours=100) for index in range(7)]
    seed(migrated_database_url, recent + stale)
    with client:
        payload = client.get(f"{DRIFT}?window_hours=24").json()
    assert payload["observed_rows"] == 5
    assert payload["window_hours"] == 24


def test_an_out_of_range_window_is_refused(client: TestClient, migrated_database_url: str) -> None:
    seed(migrated_database_url, [])
    with client:
        assert client.get(f"{DRIFT}?window_hours=0").status_code == 422
        assert client.get(f"{DRIFT}?window_hours=99999").status_code == 422


# ------------------------------------------------------------------ a real reading


@locked_only
def test_a_large_enough_window_reports_a_psi_and_a_band(
    client: TestClient, migrated_database_url: str
) -> None:
    """At demo scale the endpoint produces a real number with a bin breakdown."""

    rows = [_row(index, "0.001772338243283710") for index in range(PSI_MINIMUM_ROWS + 20)]
    seed(migrated_database_url, rows)
    with client:
        payload = client.get(DRIFT).json()

    assert payload["status"] == "ok"
    assert payload["band"] in {"none", "moderate", "significant"}
    assert math.isfinite(payload["psi"])
    assert payload["observed_rows"] == PSI_MINIMUM_ROWS + 20
    assert payload["epsilon"] == 1e-4

    assert len(payload["bins"]) == 10
    assert sum(item["observed_count"] for item in payload["bins"]) == payload["observed_rows"]
    assert sum(item["reference_count"] for item in payload["bins"]) == payload["reference_rows"]
    # Every observation is far below 0.1, so bin 0 holds all of them.
    assert payload["bins"][0]["observed_count"] == payload["observed_rows"]
    assert sum(item["contribution_share"] for item in payload["bins"]) == pytest.approx(
        1.0, abs=1e-9
    )


@locked_only
def test_the_reference_artifact_is_never_modified_by_a_reading(
    client: TestClient, migrated_database_url: str
) -> None:
    import hashlib

    before = hashlib.sha256(LOCKED.read_bytes()).hexdigest()
    seed(migrated_database_url, [_row(index, "0.001772338243283710") for index in range(5)])
    with client:
        client.get(DRIFT)
    assert hashlib.sha256(LOCKED.read_bytes()).hexdigest() == before


def test_a_missing_reference_degrades_plainly(
    integration_settings: Settings, migrated_database_url: str, tmp_path: Path
) -> None:
    seed(migrated_database_url, [])
    settings = integration_settings.model_copy(
        update={"evaluation_artifact_path": tmp_path / "absent.json"}
    )
    with TestClient(create_app(settings)) as client:
        response = client.get(DRIFT)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "reference_unavailable"
    assert payload["psi"] is None
    assert payload["bins"] == []


# ------------------------------------------------------------------ privacy


@locked_only
def test_the_drift_response_carries_no_identifier_of_any_kind(
    client: TestClient, migrated_database_url: str
) -> None:
    """Drift is aggregate by construction: there is no field able to hold a token."""

    import re

    seed(migrated_database_url, [_row(index, "0.001772338243283710") for index in range(5)])
    with client:
        payload = client.get(DRIFT).json()

    rendered = json.dumps(payload)
    assert re.search(r"(evt|mrc|chk|cus|dev|net|ses|pmt)_[0-9a-f]{32}", rendered) is None
    for forbidden in ("email", "card", "cvv", "vpa", "ip_address", "@"):
        assert forbidden not in rendered.casefold()

    assert set(payload) == {
        "status",
        "psi",
        "band",
        "observed_rows",
        "minimum_rows",
        "window_hours",
        "epsilon",
        "reference_rows",
        "reference_model_id",
        "reference_evaluation_id",
        "bins",
        "note",
    }


@locked_only
def test_the_response_states_that_drift_is_informational(
    client: TestClient, migrated_database_url: str
) -> None:
    """The claim that drift cannot influence a decision is carried in the payload itself."""

    seed(migrated_database_url, [])
    with client:
        note: str = client.get(DRIFT).json()["note"]
    assert "informational" in note.casefold()
    assert "no path to any decision" in note.casefold()
