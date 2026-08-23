"""Typed Gemini adapter.

Deliberately a thin httpx client rather than the official ``google-genai`` SDK. The SDK is the
current official package, but it brings six new runtime dependencies for a single POST, retries
internally through tenacity (this path must never retry silently), auto-discovers an ambient
``GEMINI_API_KEY``/``GOOGLE_API_KEY`` (RiskLoom keeps every secret explicit, prefixed and wrapped
in ``SecretStr``), and raises exceptions that can carry upstream response bodies, which the project
forbids logging or returning anywhere.

The wire format is confined to this module behind :class:`GeminiClientProtocol`, so replacing it --
with the SDK, or with a future API shape -- is a contained change.

Errors are stable short identities. No upstream body, header or key ever appears in an exception,
a log line or a stored row.
"""

import json
from types import TracebackType
from typing import Any, Protocol

import httpx

from riskloom.core.config import Settings
from riskloom.explanations import prompt as prompt_module
from riskloom.explanations.schemas import ExplanationInput, LlmExplanation

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com"
GENERATE_PATH = "/v1beta/interactions"
MAX_RESPONSE_BYTES = 32_768


class GeminiError(Exception):
    """Safe internal error whose message never includes an upstream body or credentials."""


class GeminiClientProtocol(Protocol):
    """What the service layer depends on. Tests substitute their own implementation."""

    async def explain(self, payload: ExplanationInput) -> LlmExplanation: ...


class GeminiClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.gemini_api_key is None:
            raise GeminiError("gemini_not_configured")
        self._model = settings.gemini_model
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=GEMINI_API_BASE_URL,
            timeout=httpx.Timeout(settings.gemini_timeout_seconds, connect=3.0),
            # No ambient proxy or certificate configuration, matching the Razorpay adapter.
            trust_env=False,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": settings.gemini_api_key.get_secret_value(),
            },
        )

    def _request_body(self, payload: ExplanationInput) -> dict[str, Any]:
        return {
            "model": self._model,
            "input": prompt_module.build(payload),
            "system_instruction": prompt_module.SYSTEM_INSTRUCTION,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": prompt_module.response_schema(),
            },
        }

    async def explain(self, payload: ExplanationInput) -> LlmExplanation:
        """One call. No retry, by design: a retry is a fresh human-initiated request."""

        try:
            response = await self._client.post(GENERATE_PATH, json=self._request_body(payload))
        except httpx.TimeoutException as exc:
            raise GeminiError("gemini_timeout") from exc
        except httpx.RequestError as exc:
            raise GeminiError("gemini_unavailable") from exc

        if response.is_error:
            raise GeminiError("gemini_rejected")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise GeminiError("gemini_invalid_response")

        try:
            text = _extract_text(response.json())
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GeminiError("gemini_invalid_response") from exc

        try:
            return LlmExplanation.model_validate(json.loads(text))
        except (ValueError, TypeError) as exc:
            raise GeminiError("gemini_invalid_response") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GeminiClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()


def _extract_text(body: Any) -> str:
    """Pull the JSON payload out of a response envelope.

    The Interactions API returns a ``steps`` list in which a ``thought`` step precedes the
    ``model_output`` step that carries the reply, so the answer is not simply the first element.
    This shape was confirmed against the live API rather than inferred from documentation. The
    older ``output`` and ``generateContent`` shapes are tolerated as fallbacks because they still
    exist upstream.
    """

    if not isinstance(body, dict):
        raise ValueError("unexpected envelope")

    steps = body.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            content = step.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return str(part["text"])

    output = body.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            return str(part["text"])

    candidates = body.get("candidates")
    if isinstance(candidates, list) and candidates:
        parts = candidates[0]["content"]["parts"]
        return str(parts[0]["text"])

    raise ValueError("unexpected envelope")
