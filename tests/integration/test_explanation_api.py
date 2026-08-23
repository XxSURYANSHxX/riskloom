"""Explanation API: eligibility, fail-safe, spend caps, and ledger immutability.

Every LLM call here is faked. A guard test in this file proves no real client can be constructed
under test settings, so the automated suite cannot reach Google even by mistake.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import riskloom.main as main_module
from riskloom.core.config import Settings
from riskloom.db.models import ReviewItem, RiskDecision
from riskloom.explanations.client import GeminiError
from riskloom.explanations.sanitizer import FORBIDDEN_KEYS, FORBIDDEN_SUBSTRINGS, TOKEN_PATTERN
from riskloom.explanations.schemas import ExplanationInput, FactorCode, LlmExplanation
from riskloom.main import create_app

pytestmark = pytest.mark.integration

THRESHOLD = Decimal("0.003386294915518273")
SHARED_DEVICE = "dev_" + "a" * 32
SHARED_NETWORK = "net_" + "b" * 32

GOOD = LlmExplanation(
    summary="This checkout was denied because calibrated risk exceeded the locked threshold.",
    factors=[FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD, FactorCode.DEVICE_REUSE],
    caveat="Derived from stored aggregates only.",
)


class FakeGemini:
    """Records what it was asked and returns whatever the test scripted."""

    def __init__(
        self,
        result: LlmExplanation | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else GOOD
        self.error = error
        self.calls: list[ExplanationInput] = []

    async def explain(self, payload: ExplanationInput) -> LlmExplanation:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.result

    async def close(self) -> None:
        return None


def _row(index: int, action: str, *, probability: str, status: str = "final") -> RiskDecision:
    now = datetime.now(UTC) - timedelta(seconds=60 - index)
    return RiskDecision(
        id=uuid4(),
        event_id=f"evt_{index:032x}",
        merchant_id=f"mrc_{1:032x}",
        checkout_id=f"chk_{index:032x}",
        session_token=f"ses_{index:032x}",
        payment_instrument_token=f"pmt_{index:032x}",
        customer_token=None,
        device_token=SHARED_DEVICE,
        network_token=SHARED_NETWORK,
        amount_subunits=25_000,
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
        razorpay_order_id=None,
        status=status,
    )


DENY = 0
ALLOW = 1
REVIEW = 2
PENDING_DENY = 3


@pytest.fixture
def seeded(migrated_database_url: str) -> list[RiskDecision]:
    rows = [
        _row(1, "deny", probability="0.007053679692244301"),
        _row(2, "allow", probability="0.001772338243283710"),
        _row(3, "review", probability="0.001772338243283710"),
        # Unreachable through preflight -- see the pending-eligibility test below.
        _row(4, "deny", probability="0.007053679692244301", status="pending"),
    ]

    async def seed() -> None:
        engine = create_async_engine(migrated_database_url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "TRUNCATE risk_decision_explanations, review_items, risk_decisions "
                        "RESTART IDENTITY CASCADE"
                    )
                )
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


def build(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeGemini | None = None,
    *,
    budget: int = 5,
    configured: bool = True,
) -> TestClient:
    client = fake or FakeGemini()
    monkeypatch.setattr(
        main_module,
        "GeminiClient",
        lambda *_a, **_k: client if configured else None,
    )
    tuned = settings.model_copy(
        update={
            "gemini_max_calls_per_process": budget,
            "gemini_api_key": _key() if configured else None,
        }
    )
    return TestClient(create_app(tuned))


def _key() -> Any:
    from pydantic import SecretStr

    return SecretStr("synthetic-test-key")


def path_for(row: RiskDecision) -> str:
    return f"/api/v1/dashboard/decisions/{row.id}/explanation"


async def _ledger_snapshot(url: str) -> list[tuple[Any, ...]]:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            decisions = (
                await connection.execute(
                    text(
                        "SELECT id, risk_decision, action, calibrated_probability, "
                        "decision_threshold, status, razorpay_order_id, fail_safe_reason "
                        "FROM risk_decisions ORDER BY event_id"
                    )
                )
            ).all()
            reviews = (
                await connection.execute(
                    text("SELECT id, status FROM review_items ORDER BY created_at")
                )
            ).all()
        return [tuple(r) for r in decisions] + [tuple(r) for r in reviews]
    finally:
        await engine.dispose()


# ------------------------------------------------------------------ eligibility


def test_a_final_deny_generates_an_explanation(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        response = client.post(path_for(seeded[DENY]))
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["summary"] == GOOD.summary
    # Factor text is RiskLoom's own rendering, never model prose.
    assert body["factors"] == [
        "risk at or above the locked threshold",
        "device reused across decisions",
    ]
    assert len(fake.calls) == 1


def test_an_allow_is_not_eligible(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        assert client.post(path_for(seeded[ALLOW])).status_code == 422
    assert fake.calls == []


def test_a_review_is_not_eligible(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REVIEW is an operational fail-safe here, never a risk band, so nothing to explain."""

    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        assert client.post(path_for(seeded[REVIEW])).status_code == 422
    assert fake.calls == []


# Flaked once (2026-08-23) in a combined run whose wall-time was ~7x normal, and has passed every
# run since. A read found no timing dependency here: eligibility compares two stored strings and
# raises before any context query, so this is the cheapest path in the file. Suspected DB
# contention rather than a defect in the test; recorded so a future session does not start cold.
def test_a_pending_row_flagged_deny_is_not_eligible(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state preflight cannot produce, constructed directly through the ORM.

    ``risk_decision`` and ``status='final'`` are written in one transaction, so this row cannot
    arise today. The guard exists because this predicate is exactly where that assumption would
    silently break under a future refactor.
    """

    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        assert client.post(path_for(seeded[PENDING_DENY])).status_code == 422
    assert fake.calls == []


def test_an_unknown_decision_is_a_clean_404(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    with build(integration_settings, monkeypatch) as client:
        response = client.post(f"/api/v1/dashboard/decisions/{uuid4()}/explanation")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ------------------------------------------------------------------ input contract


def test_only_allowlisted_facts_reach_the_client(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No raw token and no unexpected field, asserted on a row whose tokens are all populated."""

    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        client.post(path_for(seeded[DENY]))

    payload = fake.calls[0]
    assert isinstance(payload, ExplanationInput)
    serialised = payload.model_dump_json()
    assert TOKEN_PATTERN.search(serialised) is None
    for token in (SHARED_DEVICE, SHARED_NETWORK, seeded[DENY].event_id, seeded[DENY].merchant_id):
        assert token not in serialised
    assert set(json.loads(serialised)) == {
        "calibrated_probability",
        "decision_threshold",
        "probability_exceeds_threshold",
        "risk_decision",
        "action",
        "fail_safe_reason",
        "amount_subunits",
        "currency",
        "channel",
        "context",
    }


def test_the_probability_sent_matches_the_ledger_exactly(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        client.post(path_for(seeded[DENY]))
    assert fake.calls[0].calibrated_probability == "0.007053679692244301"
    assert fake.calls[0].decision_threshold == "0.003386294915518273"


# ------------------------------------------------------------------ fail-safe


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (GeminiError("gemini_timeout"), "gemini_timeout"),
        (GeminiError("gemini_unavailable"), "gemini_unavailable"),
        (GeminiError("gemini_rejected"), "gemini_rejected"),
        (GeminiError("gemini_invalid_response"), "gemini_invalid_response"),
    ],
)
def test_an_upstream_failure_stores_a_failed_state_and_leaves_the_ledger_alone(
    integration_settings: Settings,
    seeded: list[RiskDecision],
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
    error: Exception,
    reason: str,
) -> None:
    before = asyncio.run(_ledger_snapshot(migrated_database_url))
    with build(integration_settings, monkeypatch, FakeGemini(error=error)) as client:
        response = client.post(path_for(seeded[DENY]))

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "failed"
    assert body["failure_reason"] == reason
    assert body["summary"] is None
    assert asyncio.run(_ledger_snapshot(migrated_database_url)) == before


def test_a_rejected_response_is_never_stored_as_an_explanation(
    integration_settings: Settings,
    seeded: list[RiskDecision],
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> None:
    """A fabricated figure must reach neither the database nor the client."""

    hostile = LlmExplanation(
        summary="Denied after this device produced 7 decisions above the locked threshold.",
        factors=[FactorCode.DEVICE_REUSE],
        caveat="",
    )
    before = asyncio.run(_ledger_snapshot(migrated_database_url))
    with build(integration_settings, monkeypatch, FakeGemini(result=hostile)) as client:
        response = client.post(path_for(seeded[DENY]))

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "rejected"
    assert body["failure_reason"] == "unsupported_number"
    assert body["summary"] is None
    assert "7 decisions" not in json.dumps(body)
    assert asyncio.run(_ledger_snapshot(migrated_database_url)) == before


def test_a_token_bearing_response_is_rejected_and_never_stored(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    hostile = LlmExplanation(
        summary=f"Denied because {SHARED_DEVICE} exceeded the locked threshold repeatedly.",
        factors=[FactorCode.DEVICE_REUSE],
        caveat="",
    )
    with build(integration_settings, monkeypatch, FakeGemini(result=hostile)) as client:
        response = client.post(path_for(seeded[DENY]))
        body = response.json()
        assert body["status"] == "rejected"
        assert body["failure_reason"] == "forbidden_content"
        assert SHARED_DEVICE not in json.dumps(body)
        # And it stays absent on re-read.
        stored = client.get(path_for(seeded[DENY])).json()
    assert SHARED_DEVICE not in json.dumps(stored)


def test_failed_and_rejected_are_distinct_states(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One is the model not answering; the other is the model answering untrustworthily."""

    with build(
        integration_settings, monkeypatch, FakeGemini(error=GeminiError("gemini_timeout"))
    ) as client:
        first = client.post(path_for(seeded[DENY])).json()
    hostile = LlmExplanation(
        summary="Denied after this device produced 7 decisions above the locked threshold.",
        factors=[FactorCode.DEVICE_REUSE],
        caveat="",
    )
    with build(integration_settings, monkeypatch, FakeGemini(result=hostile)) as client:
        second = client.post(path_for(seeded[DENY])).json()
    assert first["status"] == "failed"
    assert second["status"] == "rejected"


# ------------------------------------------------------------------ spend caps


def test_the_process_budget_stops_further_calls(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini(error=GeminiError("gemini_timeout"))
    with build(integration_settings, monkeypatch, fake, budget=1) as client:
        assert client.post(path_for(seeded[DENY])).status_code == 202
        second = client.post(path_for(seeded[DENY]))
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "budget_exhausted"
    # The budget counts attempts, so exactly one outbound call happened.
    assert len(fake.calls) == 1


def test_a_zero_budget_makes_no_call_at_all(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake, budget=0) as client:
        assert client.post(path_for(seeded[DENY])).status_code == 429
    assert fake.calls == []


def test_attempts_per_decision_are_capped(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini(error=GeminiError("gemini_timeout"))
    with build(integration_settings, monkeypatch, fake, budget=10) as client:
        for _ in range(3):
            assert client.post(path_for(seeded[DENY])).status_code == 202
        exhausted = client.post(path_for(seeded[DENY]))
    assert exhausted.status_code == 409
    assert exhausted.json()["error"]["code"] == "attempts_exhausted"
    assert len(fake.calls) == 3


def test_a_ready_explanation_is_returned_without_a_second_call(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake, budget=10) as client:
        assert client.post(path_for(seeded[DENY])).status_code == 201
        repeat = client.post(path_for(seeded[DENY]))
    assert repeat.status_code == 201
    assert repeat.json()["status"] == "ready"
    assert len(fake.calls) == 1


def test_generation_is_refused_when_no_key_is_configured(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    with build(integration_settings, monkeypatch, configured=False) as client:
        response = client.post(path_for(seeded[DENY]))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_configured"


# ------------------------------------------------------------------ read + surface


def test_reading_before_generation_is_a_clean_404(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    with build(integration_settings, monkeypatch) as client:
        response = client.get(path_for(seeded[DENY]))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_the_explanation_path_accepts_only_get_and_post(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narrow exception to Day 7's read-only rule, bounded to two verbs."""

    with build(integration_settings, monkeypatch) as client:
        for method in ("PUT", "PATCH", "DELETE", "OPTIONS"):
            assert client.request(method, path_for(seeded[DENY])).status_code == 405


def test_the_explanation_response_carries_no_pii(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    with build(integration_settings, monkeypatch) as client:
        payload = client.post(path_for(seeded[DENY])).json()

    assert not set(payload) & FORBIDDEN_KEYS
    rendered = json.dumps(payload).casefold()
    for fragment in FORBIDDEN_SUBSTRINGS:
        assert fragment not in rendered, fragment
    assert TOKEN_PATTERN.search(json.dumps(payload)) is None
    assert set(payload) == {
        "status",
        "summary",
        "factors",
        "factor_codes",
        "caveat",
        "failure_reason",
        "model_name",
        "prompt_version",
        "attempt_number",
        "attempts_remaining",
        "created_at",
    }


def test_the_stored_row_records_the_exact_input_digest(
    integration_settings: Settings,
    seeded: list[RiskDecision],
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> None:
    """Audit property: which facts were sent is provable after the fact."""

    fake = FakeGemini()
    with build(integration_settings, monkeypatch, fake) as client:
        client.post(path_for(seeded[DENY]))

    from riskloom.services.explanations import input_digest

    expected = input_digest(fake.calls[0])

    async def read() -> tuple[str, str]:
        engine = create_async_engine(migrated_database_url)
        try:
            async with engine.begin() as connection:
                row = (
                    await connection.execute(
                        text(
                            "SELECT input_digest, status FROM risk_decision_explanations "
                            "WHERE risk_decision_id = :rid"
                        ),
                        {"rid": str(seeded[DENY].id)},
                    )
                ).one()
            return str(row[0]), str(row[1])
        finally:
            await engine.dispose()

    digest, status = asyncio.run(read())
    assert digest == expected
    assert status == "ready"


# ------------------------------------------------------------------ real-call guard


def test_test_settings_never_carry_a_real_key(integration_settings: Settings) -> None:
    """The automated suite cannot reach Google: there is no key to reach it with."""

    assert integration_settings.gemini_api_key is None


def test_every_real_client_construction_is_transport_mocked() -> None:
    """No automated test may reach Google.

    Constructing the real class is allowed, because ``respx`` replaces the httpx transport and
    nothing leaves the process. What must never happen is constructing one *without* that
    interception, so the property asserted is the pairing, not the absence.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("test_*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        if "GeminiClient(" not in source:
            continue
        assert "import respx" in source, path
        assert "@respx.mock" in source, path


def test_no_test_module_carries_a_google_api_key() -> None:
    """A Google API key has a distinctive prefix. None may be committed in a fixture.

    The needle is assembled at runtime so this file does not itself contain the literal it hunts
    for, which would make the check pass or fail on its own source rather than on the fixtures.
    """

    from pathlib import Path

    needle = "AI" + "za"
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        assert needle not in path.read_text(encoding="utf-8"), path


def test_the_decision_ledger_is_untouched_by_a_full_generation_cycle(
    integration_settings: Settings,
    seeded: list[RiskDecision],
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> None:
    """Success, failure and rejection in one process, then byte-equality on the ledger."""

    before = asyncio.run(_ledger_snapshot(migrated_database_url))
    hostile = LlmExplanation(
        summary="Denied after this device produced 7 decisions above the locked threshold.",
        factors=[FactorCode.DEVICE_REUSE],
        caveat="",
    )
    with build(
        integration_settings, monkeypatch, FakeGemini(error=GeminiError("gemini_timeout")), budget=9
    ) as client:
        client.post(path_for(seeded[DENY]))
    with build(integration_settings, monkeypatch, FakeGemini(result=hostile), budget=9) as client:
        client.post(path_for(seeded[DENY]))
    with build(integration_settings, monkeypatch, FakeGemini(), budget=9) as client:
        client.post(path_for(seeded[DENY]))

    assert asyncio.run(_ledger_snapshot(migrated_database_url)) == before


def test_a_decision_id_that_is_not_a_uuid_is_refused(
    integration_settings: Settings, seeded: list[RiskDecision], monkeypatch: pytest.MonkeyPatch
) -> None:
    with build(integration_settings, monkeypatch) as client:
        assert client.post("/api/v1/dashboard/decisions/not-a-uuid/explanation").status_code == 422


def test_uuid_round_trips_without_mutation(seeded: list[RiskDecision]) -> None:
    assert isinstance(seeded[DENY].id, UUID)
