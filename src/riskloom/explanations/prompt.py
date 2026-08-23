"""Prompt construction and the response schema sent to the model.

The prompt carries no caller-supplied text. It is built entirely from the typed
:class:`ExplanationInput`, whose every field is a number, a bool or a closed enum, so there is no
position in the rendered prompt where a user could place an instruction.
"""

import json
from typing import Any

from riskloom.explanations import factors
from riskloom.explanations.schemas import (
    MAX_CAVEAT_CHARS,
    MAX_FACTORS,
    MAX_SUMMARY_CHARS,
    ExplanationInput,
    FactorCode,
)

PROMPT_VERSION = "1"

SYSTEM_INSTRUCTION = """\
You write one short explanation of a payment risk decision that has ALREADY been made by a locked \
statistical model. You are not deciding anything. Do not suggest an action, an \
alternative outcome, or any change to the decision.

Rules you must follow exactly:
1. Use ONLY the figures given to you. Never introduce a number that is not supplied.
2. Quote any supplied figure VERBATIM, at full precision. Never round, never truncate, never \
convert to a percentage or any other derived form. Writing 0.0071 for 0.007053679692244301 is a \
serious error.
3. Prefer qualitative wording for the probability and threshold, for example "exceeded the locked \
threshold". Both exact values are already displayed to the reader, so you do not need to restate \
them. If you do restate one, it must be character-for-character identical.
4. Refer to entities generically: "this device", "this network". You are given no identifiers and \
must not invent any.
5. Choose contributing factors only from the supplied list of available codes. Do not invent a \
code and do not choose one that the figures do not support.
6. Write plain prose. No markup, no links, no lists, no headings.
7. The summary is at most two sentences and states what happened and why.
"""


def response_schema() -> dict[str, Any]:
    """JSON schema constraining the model's reply.

    ``factors`` is an enum array, so an invented factor is rejected by the provider before it ever
    reaches RiskLoom's own validation.
    """

    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": MAX_SUMMARY_CHARS},
            "factors": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_FACTORS,
                "items": {"type": "string", "enum": [code.value for code in FactorCode]},
            },
            "caveat": {"type": "string", "maxLength": MAX_CAVEAT_CHARS},
        },
        "required": ["summary", "factors", "caveat"],
    }


def render_facts(payload: ExplanationInput) -> str:
    """The facts block. Canonical JSON so an identical input renders identically."""

    supported = [code.value for code in factors.available(payload)]
    body = {
        "calibrated_probability": payload.calibrated_probability,
        "decision_threshold": payload.decision_threshold,
        "probability_exceeds_threshold": payload.probability_exceeds_threshold,
        "risk_decision": payload.risk_decision,
        "action": payload.action,
        "fail_safe_reason": payload.fail_safe_reason.value if payload.fail_safe_reason else None,
        "amount_subunits": payload.amount_subunits,
        "currency": payload.currency,
        "channel": payload.channel,
        "ledger_co_occurrence": [
            {
                "kind": aggregate.kind,
                "present": aggregate.present,
                "decision_count": aggregate.decision_count,
                "denied_count": aggregate.denied_count,
                "review_count": aggregate.review_count,
                "span_seconds": aggregate.span_seconds,
            }
            for aggregate in payload.context
        ],
        "available_factor_codes": supported,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build(payload: ExplanationInput) -> str:
    """The complete user-side prompt text."""

    return (
        "Explain this already-final risk decision.\n\n"
        "FACTS (the only information you may use):\n"
        f"{render_facts(payload)}\n\n"
        "Reply with JSON matching the required schema."
    )
