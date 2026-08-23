"""Output sanitization and grounding checks.

This module holds the single definition of the forbidden-key and forbidden-substring sets. They
were previously duplicated inside the Day 7 dashboard test; that test now imports them from here so
runtime enforcement and test assertion cannot drift apart.

The numeral cross-check is governed by one principle:

    an exact, lossless re-rendering of a supplied value is permitted;
    a lossy approximation of it is not.

That is what makes the rule coherent across field types instead of a pile of special cases. 25,000
subunits may be written ``250.00`` because dividing by 100 loses nothing. 201 seconds may be
written as ``3`` and ``21`` because 3m21s is exactly 201s. A probability of
``0.007053679692244301`` may be written verbatim or as its exact percentage
``0.7053679692244301`` -- but not as ``0.0071`` (rounded), ``0.71`` (rounded percentage) or
``0.0070`` (truncated). Truncation is rejected for the same reason rounding is: both are lossy, and
the project bans display rounding of these values everywhere else precisely because a difference
far below any sane tolerance flips the discrete decision for an entire tie cluster.

Known limitation, stated rather than hidden: numerals are checked, and cardinal number words from
two upward are checked, but "one" and "a" are not, because they are overwhelmingly idiomatic in
English prose and checking them would reject far more true statements than false ones.
"""

import re
from decimal import Decimal, InvalidOperation

from riskloom.explanations import factors
from riskloom.explanations.schemas import (
    MAX_CAVEAT_CHARS,
    MAX_SUMMARY_CHARS,
    ExplanationInput,
    ExplanationRejected,
    LlmExplanation,
)

TOKEN_PATTERN = re.compile(r"(evt|mrc|chk|cus|dev|net|ses|pmt)_[0-9a-f]{32}")

# Exact leaf-key names that must never appear in any payload RiskLoom emits.
FORBIDDEN_KEYS = frozenset(
    {
        "email",
        "contact",
        "phone",
        "card",
        "card_number",
        "pan",
        "cvv",
        "expiry",
        "ip_address",
        "vpa",
        "notes",
        "description",
        "acquirer",
        "token_id",
        "cardholder",
        "name",
    }
)

# Substrings scanned across a whole serialised body. Deliberately only unambiguous ones: short
# fragments like "pan" occur inside legitimate keys such as "span_seconds".
FORBIDDEN_SUBSTRINGS = (
    "email",
    "cardholder",
    "card_number",
    "ip_address",
    "cvv",
    "@",
    "acquirer",
    "token_id",
)

# Additional patterns that apply only to free prose returned by a model, where the dashboard's
# structural allowlist offers no protection.
_MARKUP_PATTERNS = (
    re.compile(r"<\s*[a-zA-Z/!]"),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"data\s*:\s*[a-z]+/", re.IGNORECASE),
    re.compile(r"&#\d"),
)
# The digit-run shapes deliberately refuse to match a run sitting inside a decimal number. A
# full-precision probability such as 0.007053679692244301 carries an eighteen-digit fractional run
# and would otherwise be flagged as a card number -- rejecting exactly the value this contract most
# needs to permit. A real card number is unaffected, and is caught twice over regardless: it is not
# a supported numeral either.
_PII_SHAPES = (
    re.compile(r"(?<![\d.,])[0-9]{13,19}(?![\d.,])"),  # card-number-shaped run of digits
    re.compile(r"(?<![\d.,])\+?\d[\d\s-]{8,}\d(?![\d.,])"),  # phone-shaped
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+"),  # address-shaped
)

_NUMERAL_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CURRENCY_SHAPE = re.compile(r"\b[A-Z]{3}\b")
_NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "hundred": 100,
    "thousand": 1000,
}
_CONTRADICTION_WORDS = ("approved", "allowed")

MAX_SERIALISED_BYTES = 4_096


def normalise(value: str) -> str:
    """Canonical form of a numeric literal, so equal values compare equal.

    Strips grouping commas and trailing fractional zeros. ``250.00`` and ``250`` normalise to the
    same string; ``0.0070`` normalises to ``0.007``, which is *not* the same as the full-precision
    value and is therefore correctly rejected.
    """

    text = value.replace(",", "").strip()
    if "." in text:
        whole, _, fraction = text.partition(".")
        fraction = fraction.rstrip("0")
        text = f"{whole}.{fraction}" if fraction else whole
    whole, dot, fraction = text.partition(".")
    whole = whole.lstrip("0") or "0"
    return f"{whole}{dot}{fraction}" if dot else whole


def _duration_parts(seconds: int) -> list[str]:
    """Exact components of a rendered duration. ``201`` -> 3m 21s -> ['3', '21']."""

    parts: list[str] = [str(seconds)]
    if seconds >= 60:
        parts.extend([str(seconds // 60), str(seconds % 60)])
    if seconds >= 3_600:
        parts.extend([str(seconds // 3_600), str((seconds % 3_600) // 60)])
    return parts


def permitted_numerals(payload: ExplanationInput) -> set[str]:
    """Every numeral the model is allowed to write, in normalised form.

    Membership is exact-rendering only. Nothing rounded and nothing truncated is added here, which
    is the whole point of the check.
    """

    allowed: set[str] = set()

    def add(value: str) -> None:
        allowed.add(normalise(value))

    add(str(payload.amount_subunits))
    major = Decimal(payload.amount_subunits) / Decimal(100)
    add(format(major, "f"))

    for text in (payload.calibrated_probability, payload.decision_threshold):
        add(text)
        try:
            add(format(Decimal(text) * 100, "f"))
        except InvalidOperation:  # pragma: no cover - schema constrains these to numeric strings
            continue

    for aggregate in payload.context:
        add(str(aggregate.decision_count))
        add(str(aggregate.denied_count))
        add(str(aggregate.review_count))
        if aggregate.span_seconds is not None:
            for part in _duration_parts(aggregate.span_seconds):
                add(part)

    return allowed


def _check_numerals(text: str, allowed: set[str]) -> None:
    for match in _NUMERAL_PATTERN.finditer(text):
        if normalise(match.group(0)) not in allowed:
            raise ExplanationRejected("unsupported_number")
    lowered = text.casefold()
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered) and str(value) not in allowed:
            raise ExplanationRejected("unsupported_number")


def _check_forbidden(text: str) -> None:
    if TOKEN_PATTERN.search(text):
        raise ExplanationRejected("forbidden_content")
    lowered = text.casefold()
    for fragment in FORBIDDEN_SUBSTRINGS:
        if fragment in lowered:
            raise ExplanationRejected("forbidden_content")
    for pattern in (*_MARKUP_PATTERNS, *_PII_SHAPES):
        if pattern.search(text):
            raise ExplanationRejected("forbidden_content")


def verify(explanation: LlmExplanation, payload: ExplanationInput) -> LlmExplanation:
    """Run every grounding and safety check. Returns the explanation or raises.

    Order matters: cheap structural checks run before the numeral walk so a hostile response is
    refused as early as possible.
    """

    prose = (explanation.summary, explanation.caveat)

    if len(explanation.summary) > MAX_SUMMARY_CHARS or len(explanation.caveat) > MAX_CAVEAT_CHARS:
        raise ExplanationRejected("malformed_response")
    if sum(len(part.encode("utf-8")) for part in prose) > MAX_SERIALISED_BYTES:
        raise ExplanationRejected("malformed_response")

    if len(set(explanation.factors)) != len(explanation.factors):
        raise ExplanationRejected("unsupported_factor")
    if factors.unsupported(explanation.factors, payload):
        raise ExplanationRejected("unsupported_factor")

    for part in prose:
        _check_forbidden(part)

    for part in prose:
        for currency in _CURRENCY_SHAPE.findall(part):
            if currency != payload.currency:
                raise ExplanationRejected("unsupported_number")

    if payload.action == "deny":
        for part in prose:
            lowered = part.casefold()
            for word in _CONTRADICTION_WORDS:
                if re.search(rf"\b{word}\b", lowered):
                    raise ExplanationRejected("contradicts_decision")

    allowed = permitted_numerals(payload)
    for part in prose:
        _check_numerals(part, allowed)

    return explanation
