"""Dashboard API: read-only, faithful to the ledger, and free of PII."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from riskloom.core.config import Settings
from riskloom.db.models import ReviewItem, RiskDecision

# Imported rather than redefined: these are the sets the runtime sanitizer enforces, so a
# change to one cannot silently diverge from the other.
from riskloom.explanations.sanitizer import FORBIDDEN_KEYS, FORBIDDEN_SUBSTRINGS
from riskloom.main import create_app

pytestmark = pytest.mark.integration

ENDPOINTS = (
    "/api/v1/dashboard/summary",
    "/api/v1/dashboard/decisions",
    "/api/v1/dashboard/coordination",
    "/api/v1/dashboard/model",
)
NON_GET = ("POST", "PUT", "PATCH", "DELETE", "OPTIONS")

THRESHOLD = Decimal("0.003386294915518273")
SHARED_DEVICE = "dev_" + "a" * 32
SHARED_NETWORK = "net_" + "b" * 32


def _row(index: int, action: str, *, shared: bool, probability: str, order: str | None = None):
    # Anchored to now, not a fixed calendar date: the coordination graph only considers a
    # recent window, so fixed timestamps would age out of it and the graph would be empty.
    now = datetime.now(UTC) - timedelta(seconds=60 - index)
    return RiskDecision(
        id=uuid4(),
        event_id=f"evt_{index:032x}",
        merchant_id=f"mrc_{1:032x}",
        checkout_id=f"chk_{index:032x}",
        session_token=f"ses_{index:032x}",
        payment_instrument_token=f"pmt_{index:032x}",
        customer_token=None,
        device_token=SHARED_DEVICE if shared else f"dev_{index:032x}",
        network_token=SHARED_NETWORK if shared else f"net_{index:032x}",
        amount_subunits=25_000 + index,
        currency="INR",
        channel="web",
        occurred_at=now,
        calibrated_probability=Decimal(probability),
        decision_threshold=THRESHOLD,
        risk_decision="deny" if action == "deny" else "allow",
        action=action,
        fail_safe_reason="order_creation_failed" if action == "review" else None,
        model_id="d" * 64,
        feature_dataset_id="e" * 64,
        feature_schema_version="1.0.0",
        feature_engine_version="1.0.0",
        razorpay_order_id=order,
        status="final",
    )


@pytest.fixture
def seeded(migrated_database_url: str) -> list[RiskDecision]:
    rows = [
        _row(1, "review", shared=True, probability="0.001772338243283710"),
        _row(2, "review", shared=True, probability="0.001772338243283710"),
        _row(3, "review", shared=True, probability="0.003386294915518273"),
        _row(4, "deny", shared=True, probability="0.007053679692244301"),
        _row(5, "allow", shared=False, probability="0.001772338243283710", order="order_TEST01"),
    ]

    async def seed() -> None:
        engine = create_async_engine(migrated_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text("TRUNCATE review_items, risk_decisions RESTART IDENTITY CASCADE")
                )
            # expire_on_commit=False so the seeded rows stay readable after the session
            # closes; the tests compare API output against these objects directly.
            async with (
                AsyncSession(engine, expire_on_commit=False) as session,
                session.begin(),
            ):
                for row in rows:
                    session.add(row)
                    if row.action == "review":
                        session.add(
                            ReviewItem(
                                risk_decision_id=row.id,
                                merchant_id=row.merchant_id,
                                checkout_id=row.checkout_id,
                                status="pending",
                            )
                        )
        finally:
            await engine.dispose()

    asyncio.run(seed())
    return rows


@pytest.fixture
def client(integration_settings: Settings) -> TestClient:
    return TestClient(create_app(integration_settings))


# ------------------------------------------------------------------ read-only


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize("method", NON_GET)
def test_dashboard_endpoints_reject_every_non_get_method(
    client: TestClient, seeded: list[RiskDecision], path: str, method: str
) -> None:
    """Enforced by routing: the path is registered for GET only, so Starlette answers 405."""

    with client:
        assert client.request(method, path).status_code == 405


def test_decision_detail_endpoint_is_also_get_only(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    path = f"/api/v1/dashboard/decisions/{seeded[0].id}"
    with client:
        assert client.get(path).status_code == 200
        for method in NON_GET:
            assert client.request(method, path).status_code == 405


def test_static_dashboard_mount_serves_only_safe_methods(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    with client:
        assert client.get("/dashboard/").status_code == 200
        assert client.post("/dashboard/").status_code in (404, 405)


# ------------------------------------------------------------------- fidelity


def test_stream_matches_the_ledger_rows_exactly(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    """Displayed figures must equal their source of truth on the fields that matter."""

    with client:
        payload = client.get("/api/v1/dashboard/decisions?limit=50").json()

    assert payload["total"] == 5
    by_event = {d["event_id"]: d for d in payload["decisions"]}
    for row in seeded:
        shown = by_event[row.event_id]
        # Exact decimal string, never a float round-trip.
        assert shown["calibrated_probability"] == format(row.calibrated_probability, "f")
        assert shown["decision_threshold"] == format(row.decision_threshold, "f")
        assert shown["risk_decision"] == row.risk_decision
        assert shown["action"] == row.action
        assert shown["fail_safe_reason"] == row.fail_safe_reason
        assert shown["razorpay_order_id"] == row.razorpay_order_id
        assert shown["amount_subunits"] == row.amount_subunits
        assert shown["model_id"] == row.model_id


def test_probability_precision_survives_the_api(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    """The tie-cluster value must arrive intact, digit for digit.

    Row 3 carries exactly the probability that sits one unit in the last place below the locked
    threshold. If the API round-tripped it through a float this assertion would fail.
    """

    with client:
        payload = client.get("/api/v1/dashboard/decisions?limit=50").json()
    shown = {d["event_id"]: d for d in payload["decisions"]}[f"evt_{3:032x}"]
    assert shown["calibrated_probability"] == "0.003386294915518273"
    assert shown["decision_threshold"] == "0.003386294915518273"


def test_summary_counts_match_the_ledger(client: TestClient, seeded: list[RiskDecision]) -> None:
    with client:
        summary = client.get("/api/v1/dashboard/summary").json()
    assert summary["total_decisions"] == 5
    assert summary["actions"] == {"allow": 1, "review": 3, "deny": 1}
    assert summary["orders_created"] == 1
    assert summary["review_items_pending"] == 3


def test_action_filter_and_paging(client: TestClient, seeded: list[RiskDecision]) -> None:
    with client:
        review = client.get("/api/v1/dashboard/decisions?action=review").json()
        page = client.get("/api/v1/dashboard/decisions?limit=2&offset=0").json()
        rejected = client.get("/api/v1/dashboard/decisions?action=bogus")
    assert review["total"] == 3
    assert all(d["action"] == "review" for d in review["decisions"])
    assert len(page["decisions"]) == 2 and page["total"] == 5
    assert rejected.status_code == 422


def test_decision_detail_context_counts_shared_tokens(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    deny = next(row for row in seeded if row.action == "deny")
    with client:
        detail = client.get(f"/api/v1/dashboard/decisions/{deny.id}").json()

    context = {entry["kind"]: entry for entry in detail["context"]}
    # Four decisions share the device and network; one of them denied.
    assert context["device"]["decision_count"] == 4
    assert context["device"]["denied_count"] == 1
    assert context["device"]["review_count"] == 3
    assert context["network"]["decision_count"] == 4
    # The instrument is unique to this decision.
    assert context["instrument"]["decision_count"] == 1
    assert detail["review_pending"] is False


def test_unknown_decision_is_a_clean_404(client: TestClient, seeded: list[RiskDecision]) -> None:
    with client:
        response = client.get(f"/api/v1/dashboard/decisions/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "decision_not_found"


# --------------------------------------------------------------- coordination


def test_coordination_graph_reflects_the_shared_tokens(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    with client:
        graph = client.get("/api/v1/dashboard/coordination?window_seconds=86400").json()

    hubs = [n for n in graph["nodes"] if n["kind"] != "event"]
    events = [n for n in graph["nodes"] if n["kind"] == "event"]
    assert {h["kind"] for h in hubs} == {"device", "network"}
    assert all(h["degree"] == 4 for h in hubs)
    # The four burst decisions appear once each, not once per hub.
    assert len(events) == 4
    assert len(graph["edges"]) == 8
    # The lone unshared decision contributes no hub.
    assert graph["clustered_entity_count"] == 2
    assert {e["action"] for e in events} == {"review", "deny"}


def test_graph_layout_is_deterministic(client: TestClient, seeded: list[RiskDecision]) -> None:
    with client:
        first = client.get("/api/v1/dashboard/coordination?window_seconds=86400").json()
        second = client.get("/api/v1/dashboard/coordination?window_seconds=86400").json()
    assert first == second


def test_graph_excludes_entities_seen_only_once(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    """A token used once carries no coordination signal and must not become a hub."""

    with client:
        graph = client.get("/api/v1/dashboard/coordination?window_seconds=86400").json()
    labels = {n["label"] for n in graph["nodes"] if n["kind"] != "event"}
    assert labels == {SHARED_DEVICE, SHARED_NETWORK}
    assert not any(n["kind"] == "instrument" for n in graph["nodes"])


# ---------------------------------------------------------------------- PII


DECISION_FIELDS = {
    "decision_id",
    "event_id",
    "merchant_id",
    "checkout_id",
    "device_token",
    "network_token",
    "payment_instrument_token",
    "session_token",
    "customer_token",
    "amount_subunits",
    "currency",
    "channel",
    "occurred_at",
    "created_at",
    "calibrated_probability",
    "decision_threshold",
    "risk_decision",
    "action",
    "fail_safe_reason",
    "razorpay_order_id",
    "status",
    "model_id",
}


def _leaf_keys(value: Any, found: set[str] | None = None) -> set[str]:
    found = found if found is not None else set()
    if isinstance(value, dict):
        for key, nested in value.items():
            found.add(key)
            _leaf_keys(nested, found)
    elif isinstance(value, list):
        for nested in value:
            _leaf_keys(nested, found)
    return found


def test_decision_responses_expose_exactly_the_allowlisted_fields(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    with client:
        page = client.get("/api/v1/dashboard/decisions?limit=50").json()
        detail = client.get(f"/api/v1/dashboard/decisions/{seeded[0].id}").json()

    assert set(page["decisions"][0]) == DECISION_FIELDS
    assert set(detail["decision"]) == DECISION_FIELDS
    for blob in (page, detail):
        assert not _leaf_keys(blob) & FORBIDDEN_KEYS
        rendered = json.dumps(blob).casefold()
        for token in FORBIDDEN_SUBSTRINGS:
            assert token not in rendered, token


def test_every_identifier_is_a_pseudonymous_token(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    import re  # noqa: PLC0415

    pattern = re.compile(r"^(evt|mrc|chk|cus|dev|net|ses|pmt)_[0-9a-f]{32}$")
    with client:
        page = client.get("/api/v1/dashboard/decisions?limit=50").json()
    for decision in page["decisions"]:
        for field in (
            "event_id",
            "merchant_id",
            "checkout_id",
            "device_token",
            "network_token",
            "session_token",
            "payment_instrument_token",
        ):
            value = decision[field]
            if value is not None:
                assert pattern.fullmatch(value), (field, value)


MODEL_TOP_LEVEL = {
    "model_id",
    "evaluation_id",
    "row_count",
    "threshold",
    "probability",
    "reliability",
    "hard_negative_slices",
    "campaigns",
}
MODEL_ALL_KEYS = MODEL_TOP_LEVEL | {
    # threshold
    "attack_count",
    "legitimate_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "false_positive_rate",
    "false_positives_per_10000_legitimate",
    "prevalence",
    "review_count",
    "review_rate",
    "cost_units",
    "cost_per_10000_events",
    # probability
    "average_precision",
    "pr_auc_trapezoidal",
    "roc_auc",
    "brier_score",
    "log_loss",
    # reliability
    "expected_calibration_error",
    "bins",
    "lower_inclusive",
    "upper_value",
    "upper_inclusive",
    "count",
    "mean_probability",
    "attack_rate",
    # slices
    "slice_name",
    "false_positive_count",
    "false_positives_per_10000",
    # campaigns
    "campaign_count",
    "detected_campaign_count",
    "missed_campaign_count",
    "campaign_recall",
    "detection_delay_ms_minimum",
    "detection_delay_ms_median",
    "detection_delay_ms_p95",
    "detection_delay_ms_maximum",
    "flagged_per_campaign_minimum",
    "flagged_per_campaign_median",
    "flagged_per_campaign_maximum",
}
MODEL_FORBIDDEN = (
    "evt_",
    "cmp_",
    "scn_",
    "event_id",
    "campaign_id",
    "prediction",
    "predictions",
    "probabilities",
    "per_event",
    "scores",
    "histogram",
    "device",
    "network",
    "customer",
    "merchant",
    "checkout",
    "instrument",
)


@pytest.mark.skipif(
    not Path("artifacts/evaluations/development/evaluation.json").exists(),
    reason="evaluation artifact is Git-ignored and absent in this checkout",
)
def test_model_endpoint_exposes_only_aggregate_metrics(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    """Same scrutiny as /decisions, applied to the offline artifact projection.

    evaluation.json is aggregate-only by construction, but this endpoint never passes it through:
    every field is selected by name. This asserts the projection's whole key set and that nothing
    resembling an event id, campaign id or per-event prediction reaches a client.
    """

    with client:
        response = client.get("/api/v1/dashboard/model")
    assert response.status_code == 200
    payload = response.json()

    assert set(payload) == MODEL_TOP_LEVEL
    assert _leaf_keys(payload) == MODEL_ALL_KEYS

    assert not _leaf_keys(payload) & FORBIDDEN_KEYS
    rendered = json.dumps(payload).casefold()
    for token in MODEL_FORBIDDEN:
        assert token not in rendered, token
    for token in FORBIDDEN_SUBSTRINGS:
        assert token not in rendered, token

    # The only array is the fixed ten-bin reliability curve; nothing per-event.
    arrays = [k for k, v in payload.items() if isinstance(v, list)]
    assert arrays == ["hard_negative_slices"]
    assert len(payload["reliability"]["bins"]) == 10
    assert len(payload["hard_negative_slices"]) == 5


@pytest.mark.skipif(
    not Path("artifacts/evaluations/development/evaluation.json").exists(),
    reason="evaluation artifact is Git-ignored and absent in this checkout",
)
def test_model_endpoint_matches_the_artifact_values(
    client: TestClient, seeded: list[RiskDecision]
) -> None:
    source = json.loads(Path("artifacts/evaluations/development/evaluation.json").read_bytes())
    with client:
        payload = client.get("/api/v1/dashboard/model").json()

    assert payload["model_id"] == source["model_id"]
    assert payload["row_count"] == source["row_count"]
    threshold = source["metrics"]["threshold"]
    assert payload["threshold"]["recall"] == threshold["recall"]
    assert payload["threshold"]["precision"] == threshold["precision"]
    assert payload["threshold"]["cost_units"] == threshold["cost_units"]
    campaigns = source["metrics"]["campaigns"]
    assert payload["campaigns"]["campaign_count"] == campaigns["campaign_count"]
    assert payload["campaigns"]["campaign_recall"] == campaigns["campaign_recall"]


def test_model_endpoint_is_404_when_the_artifact_is_absent(
    integration_settings: Settings, seeded: list[RiskDecision], tmp_path: Path
) -> None:
    settings = integration_settings.model_copy(
        update={"evaluation_artifact_path": tmp_path / "absent.json"}
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/dashboard/model")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "evaluation_unavailable"
