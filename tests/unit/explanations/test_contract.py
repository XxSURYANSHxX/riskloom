"""Input contract, output schema, factor predicates and the client adapter."""

import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from riskloom.core.config import Settings
from riskloom.explanations import factors, prompt
from riskloom.explanations.client import (
    GEMINI_API_BASE_URL,
    GENERATE_PATH,
    GeminiClient,
    GeminiError,
)
from riskloom.explanations.sanitizer import TOKEN_PATTERN
from riskloom.explanations.schemas import (
    EntityAggregate,
    FactorCode,
    LlmExplanation,
)
from tests.unit.explanations.test_sanitizer import make_input

GOOD_BODY = {
    "summary": "This checkout was denied because calibrated risk exceeded the locked threshold.",
    "factors": ["probability_at_or_above_threshold", "device_reuse"],
    "caveat": "Based on stored aggregates only.",
}


def envelope(body: dict[str, object]) -> dict[str, object]:
    return {"output": [{"content": [{"text": json.dumps(body)}]}]}


@pytest.fixture
def settings(unit_settings: Settings) -> Settings:
    """Test settings carry no key by default; this fixture adds a synthetic one."""

    return unit_settings.model_copy(update={"gemini_api_key": SecretStr("test-key")})


# --------------------------------------------------------------------------- input contract

ALLOWED_INPUT_KEYS = {
    "calibrated_probability",
    "decision_threshold",
    "probability_exceeds_threshold",
    "risk_decision",
    "action",
    "fail_safe_reason",
    "amount_subunits",
    "currency",
    "channel",
    "ledger_co_occurrence",
    "available_factor_codes",
}
ALLOWED_AGGREGATE_KEYS = {
    "kind",
    "present",
    "decision_count",
    "denied_count",
    "review_count",
    "span_seconds",
}


def test_only_allowlisted_fields_are_ever_sent() -> None:
    """The rendered facts block is the exact payload that leaves the process."""

    body = json.loads(prompt.render_facts(make_input()))
    assert set(body) == ALLOWED_INPUT_KEYS
    for aggregate in body["ledger_co_occurrence"]:
        assert set(aggregate) == ALLOWED_AGGREGATE_KEYS


def test_no_pseudonymous_token_can_reach_the_model() -> None:
    """Not filtered out -- structurally absent. ``EntityAggregate`` has no field to hold one."""

    assert "token" not in EntityAggregate.model_fields
    rendered = prompt.build(make_input())
    assert TOKEN_PATTERN.search(rendered) is None
    for fragment in ("evt_", "dev_", "net_", "pmt_", "ses_", "mrc_", "chk_", "cus_"):
        assert fragment not in rendered


def test_the_input_schema_has_no_free_text_field() -> None:
    """The anti-injection property, asserted rather than asserted-in-prose.

    Every field is a number, a bool, or drawn from a closed set. The two string fields are
    constrained: probabilities are numeric strings and currency matches a three-letter pattern.
    """

    payload = make_input()
    for name, value in payload.model_dump().items():
        if name == "context":
            continue
        if isinstance(value, str):
            assert name in {
                "calibrated_probability",
                "decision_threshold",
                "risk_decision",
                "action",
                "channel",
                "currency",
                "fail_safe_reason",
            }, name


@pytest.mark.parametrize(
    "override",
    [
        {"currency": "inr"},
        {"currency": "RUPEES"},
        {"amount_subunits": 0},
        {"channel": "pos"},
        {"risk_decision": "review"},
    ],
)
def test_the_input_schema_rejects_out_of_contract_values(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_input(**override)


def test_an_unknown_input_field_is_refused() -> None:
    with pytest.raises(ValueError):
        make_input(device_token="dev_" + "a" * 32)


def test_rendered_facts_are_canonical_and_stable() -> None:
    payload = make_input()
    assert prompt.render_facts(payload) == prompt.render_facts(payload)
    assert '":' in prompt.render_facts(payload)  # compact separators


# --------------------------------------------------------------------------- output schema


@pytest.mark.parametrize(
    "body",
    [
        {"factors": ["device_reuse"], "caveat": ""},  # missing summary
        {"summary": "x" * 40, "caveat": ""},  # missing factors
        {"summary": "x" * 40, "factors": [], "caveat": ""},  # empty factors
        {"summary": "short", "factors": ["device_reuse"], "caveat": ""},  # too short
        {"summary": "x" * 401, "factors": ["device_reuse"], "caveat": ""},  # too long
        {"summary": "x" * 40, "factors": ["invented_code"], "caveat": ""},  # unknown code
        {"summary": "x" * 40, "factors": ["device_reuse"], "caveat": "", "extra": 1},  # unknown key
        {"summary": 42, "factors": ["device_reuse"], "caveat": ""},  # wrong type
        {"summary": "x" * 40, "factors": "device_reuse", "caveat": ""},  # wrong container
        {
            "summary": "x" * 40,
            "factors": ["device_reuse"] * 5,
            "caveat": "",
        },  # too many
    ],
)
def test_a_malformed_response_never_validates(body: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LlmExplanation.model_validate(body)


def test_the_response_schema_offers_only_known_codes() -> None:
    schema = prompt.response_schema()
    assert set(schema["properties"]["factors"]["items"]["enum"]) == {
        code.value for code in FactorCode
    }


# --------------------------------------------------------------------------- factor predicates


def test_every_factor_code_has_a_predicate_and_a_label() -> None:
    for code in FactorCode:
        assert code in factors.SUPPORT
        assert factors.render(code)


def test_each_predicate_is_true_and_false_for_some_input() -> None:
    supported = make_input(
        probability_exceeds_threshold=True,
        context=[
            EntityAggregate(
                kind=kind,
                present=True,
                decision_count=6,
                denied_count=2,
                review_count=2,
                span_seconds=120,
            )
            for kind in ("device", "network", "instrument", "merchant")
        ],
    )
    bare = make_input(
        probability_exceeds_threshold=False,
        context=[
            EntityAggregate(
                kind=kind,
                present=False,
                decision_count=0,
                denied_count=0,
                review_count=0,
                span_seconds=None,
            )
            for kind in ("device", "network", "instrument", "merchant")
        ],
    )
    for code in FactorCode:
        assert factors.is_supported(code, supported), code
        assert not factors.is_supported(code, bare), code


def test_an_absent_entity_supports_nothing() -> None:
    payload = make_input(
        context=[
            EntityAggregate(
                kind="device",
                present=False,
                decision_count=9,
                denied_count=9,
                review_count=9,
                span_seconds=1,
            )
        ]
    )
    assert not factors.is_supported(FactorCode.DEVICE_REUSE, payload)
    assert not factors.is_supported(FactorCode.PRIOR_DENIALS_ON_DEVICE, payload)


def test_the_prompt_offers_only_supported_codes() -> None:
    payload = make_input()
    offered = json.loads(prompt.render_facts(payload))["available_factor_codes"]
    assert offered == [code.value for code in factors.available(payload)]
    assert "prior_denials_on_instrument" not in offered


# --------------------------------------------------------------------------- client adapter


@respx.mock
async def test_the_client_sends_only_the_allowlisted_payload(settings: Settings) -> None:
    route = respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(200, json=envelope(GOOD_BODY))
    )
    async with GeminiClient(settings) as client:
        await client.explain(make_input())

    sent = json.loads(route.calls[0].request.content)
    assert set(sent) == {"model", "input", "system_instruction", "response_format"}
    assert sent["model"] == settings.gemini_model
    assert TOKEN_PATTERN.search(sent["input"]) is None


@respx.mock
async def test_the_api_key_travels_only_in_the_header(settings: Settings) -> None:
    route = respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(200, json=envelope(GOOD_BODY))
    )
    async with GeminiClient(settings) as client:
        await client.explain(make_input())
    request = route.calls[0].request
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "test-key" not in request.content.decode()
    assert "test-key" not in str(request.url)


@respx.mock
@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        (httpx.TimeoutException("slow"), "gemini_timeout"),
        (httpx.ConnectError("down"), "gemini_unavailable"),
    ],
)
async def test_transport_failures_map_to_safe_identities(
    settings: Settings, side_effect: Exception, expected: str
) -> None:
    respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(side_effect=side_effect)
    async with GeminiClient(settings) as client:
        with pytest.raises(GeminiError, match=expected):
            await client.explain(make_input())


@respx.mock
async def test_an_upstream_error_body_never_escapes(settings: Settings) -> None:
    secret_body = {"error": {"message": "quota for project acme-prod-42 exhausted"}}
    respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(429, json=secret_body)
    )
    async with GeminiClient(settings) as client:
        with pytest.raises(GeminiError) as caught:
            await client.explain(make_input())
    rendered = f"{caught.value!r} {caught.value.args}"
    assert str(caught.value) == "gemini_rejected"
    assert "acme-prod-42" not in rendered
    assert "quota" not in rendered


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json at all"),
        httpx.Response(200, json={"unexpected": "envelope"}),
        httpx.Response(200, json=envelope({"summary": "too short"})),
        httpx.Response(200, json={"output": [{"content": [{"text": "{not json"}]}]}),
    ],
)
async def test_an_unusable_response_maps_to_a_safe_identity(
    settings: Settings, response: httpx.Response
) -> None:
    respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(return_value=response)
    async with GeminiClient(settings) as client:
        with pytest.raises(GeminiError, match="gemini_invalid_response"):
            await client.explain(make_input())


@respx.mock
async def test_an_oversized_response_is_refused(settings: Settings) -> None:
    respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(200, content=b"x" * 40_000)
    )
    async with GeminiClient(settings) as client:
        with pytest.raises(GeminiError, match="gemini_invalid_response"):
            await client.explain(make_input())


@respx.mock
async def test_the_client_never_retries(settings: Settings) -> None:
    """A retry is a fresh, human-initiated request. The adapter must not loop on its own."""

    route = respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    async with GeminiClient(settings) as client:
        with pytest.raises(GeminiError):
            await client.explain(make_input())
    assert route.call_count == 1


async def test_the_client_refuses_to_construct_without_a_key(unit_settings: Settings) -> None:
    with pytest.raises(GeminiError, match="gemini_not_configured"):
        GeminiClient(unit_settings.model_copy(update={"gemini_api_key": None}))


@respx.mock
async def test_the_legacy_generate_content_envelope_is_also_understood(settings: Settings) -> None:
    """Both upstream shapes coexist; the adapter tolerates either."""

    legacy = {"candidates": [{"content": {"parts": [{"text": json.dumps(GOOD_BODY)}]}}]}
    respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(200, json=legacy)
    )
    async with GeminiClient(settings) as client:
        result = await client.explain(make_input())
    assert result.factors == [FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD, FactorCode.DEVICE_REUSE]


@respx.mock
async def test_the_live_interactions_envelope_is_parsed(settings: Settings) -> None:
    """The exact shape returned by the real API, captured during manual verification.

    A ``thought`` step precedes the ``model_output`` step, so the reply is not the first element.
    Pinned here because it was confirmed against the live service, not inferred from docs.
    """

    live = {
        "id": "v1_synthetic",
        "status": "completed",
        "model": "gemini-3.6-flash",
        "steps": [
            {"type": "thought", "signature": "c3ludGhldGlj"},
            {"type": "model_output", "content": [{"text": json.dumps(GOOD_BODY), "type": "text"}]},
        ],
    }
    respx.post(f"{GEMINI_API_BASE_URL}{GENERATE_PATH}").mock(
        return_value=httpx.Response(200, json=live)
    )
    async with GeminiClient(settings) as client:
        result = await client.explain(make_input())
    assert result.summary == GOOD_BODY["summary"]
    assert [code.value for code in result.factors] == GOOD_BODY["factors"]
