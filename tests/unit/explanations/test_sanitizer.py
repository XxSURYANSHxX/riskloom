"""Grounding and safety checks on model output.

The numeral tests are the centre of this file. The governing rule is that an *exact, lossless*
re-rendering of a supplied value is permitted and a *lossy approximation* of it is not -- so a
truthful-but-rounded restatement is rejected exactly as an invented figure is. That strictness is a
deliberate decision, recorded here so it cannot be softened by accident.
"""

import pytest

from riskloom.explanations.sanitizer import (
    ExplanationRejected,
    normalise,
    permitted_numerals,
    verify,
)
from riskloom.explanations.schemas import (
    EntityAggregate,
    ExplanationInput,
    FactorCode,
    LlmExplanation,
)

# The real Day 6 DENY: probability above the locked threshold, on a device carrying four decisions
# inside 3m 21s.
PROBABILITY = "0.007053679692244301"
THRESHOLD = "0.003386294915518273"


def make_input(**overrides: object) -> ExplanationInput:
    payload: dict[str, object] = {
        "calibrated_probability": PROBABILITY,
        "decision_threshold": THRESHOLD,
        "probability_exceeds_threshold": True,
        "risk_decision": "deny",
        "action": "deny",
        "fail_safe_reason": None,
        "amount_subunits": 25_000,
        "currency": "INR",
        "channel": "web",
        "context": [
            EntityAggregate(
                kind="device",
                present=True,
                decision_count=4,
                denied_count=1,
                review_count=3,
                span_seconds=201,
            ),
            EntityAggregate(
                kind="network",
                present=True,
                decision_count=4,
                denied_count=1,
                review_count=3,
                span_seconds=201,
            ),
        ],
    }
    payload.update(overrides)
    return ExplanationInput(**payload)  # type: ignore[arg-type]


def explanation(summary: str, caveat: str = "") -> LlmExplanation:
    return LlmExplanation(
        summary=summary,
        factors=[FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD, FactorCode.DEVICE_REUSE],
        caveat=caveat,
    )


# --------------------------------------------------------------------------- rounding: rejected


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Risk was 0.0071 which exceeded the locked threshold for this device.", "rounded up"),
        ("Risk was roughly 0.71% against the locked threshold for this device.", "percentage"),
        ("Risk was 0.0070 against the locked threshold on this reused device.", "truncated"),
        ("Risk reached 0.007 on this device, above the locked threshold value.", "short prefix"),
        ("Risk was about 0.00705 for this device, above the locked threshold.", "long prefix"),
    ],
)
def test_a_rounded_or_truncated_probability_is_rejected(text: str, label: str) -> None:
    """A *truthful* restatement is still rejected when it is lossy.

    This is the decision the gate was asked to make explicit: rounding is refused for the same
    reason truncation is. The locked threshold carries nineteen significant decimals and a real
    tie-cluster sits one unit in the last place below it, so an approximation of these values is
    exactly the display rounding the project bans everywhere else.
    """

    with pytest.raises(ExplanationRejected, match="unsupported_number"):
        verify(explanation(text), make_input())


def test_a_rounded_amount_is_rejected() -> None:
    with pytest.raises(ExplanationRejected, match="unsupported_number"):
        verify(
            explanation("The 260.00 INR charge on this device was denied on risk."), make_input()
        )


def test_a_rounded_duration_is_rejected() -> None:
    with pytest.raises(ExplanationRejected, match="unsupported_number"):
        verify(
            explanation("Four decisions arrived from this device within 3.35 minutes total."),
            make_input(),
        )


def test_an_invented_count_is_rejected() -> None:
    with pytest.raises(ExplanationRejected, match="unsupported_number"):
        verify(
            explanation("This device carried 7 decisions above the locked threshold."), make_input()
        )


def test_an_invented_count_written_as_a_word_is_rejected() -> None:
    with pytest.raises(ExplanationRejected, match="unsupported_number"):
        verify(
            explanation("This device carried seven decisions above the locked threshold."),
            make_input(),
        )


# --------------------------------------------------------------------------- lossless: accepted


def test_the_verbatim_full_precision_probability_is_accepted() -> None:
    """The exact stored string, character for character."""

    text = f"Risk of {PROBABILITY} exceeded the locked threshold of {THRESHOLD} on this device."
    assert verify(explanation(text), make_input()).summary == text


def test_the_exact_percentage_conversion_is_accepted() -> None:
    """Multiplying by 100 loses nothing, so the exact percentage is a lossless re-rendering."""

    text = "Risk of 0.7053679692244301% exceeded the locked threshold on this reused device."
    assert verify(explanation(text), make_input()).summary == text


@pytest.mark.parametrize("rendering", ["25000", "25,000", "250.00", "250"])
def test_every_exact_amount_rendering_is_accepted(rendering: str) -> None:
    """Major units are exact: 25000/100 loses nothing."""

    text = f"The {rendering} INR attempt on this device exceeded the locked threshold."
    assert verify(explanation(text), make_input()).summary == text


@pytest.mark.parametrize("rendering", ["201 seconds", "3 minutes and 21 seconds"])
def test_every_exact_duration_rendering_is_accepted(rendering: str) -> None:
    """3m21s is exactly 201s, so both renderings are lossless."""

    text = f"Four decisions came from this device in {rendering}, above the locked threshold."
    assert verify(explanation(text), make_input()).summary == text


def test_qualitative_phrasing_with_no_figures_is_accepted() -> None:
    """The shape the prompt actually asks for."""

    text = "This checkout was denied because its calibrated risk exceeded the locked threshold."
    assert verify(explanation(text), make_input()).summary == text


def test_the_rule_is_identical_across_field_types() -> None:
    """One rule, not one per field: every exact rendering passes, every lossy one fails."""

    payload = make_input()
    allowed = permitted_numerals(payload)
    for exact in ("25000", "250.00", "250", "201", "3", "21", "4", "1", PROBABILITY):
        assert normalise(exact) in allowed, exact
    for lossy in ("0.0071", "0.71", "0.0070", "260.00", "3.35", "200", "7"):
        assert normalise(lossy) not in allowed, lossy


# --------------------------------------------------------------------------- adversarial content


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Denied after reuse by dev_" + "a" * 32 + " on this network.", "forbidden_content"),
        ("Denied; contact the payer at fraud.team@example.com about this.", "forbidden_content"),
        (
            "<script>alert(1)</script> This device was denied above the threshold.",
            "forbidden_content",
        ),
        ("Denied on this device. See javascript:void(0) for the full detail.", "forbidden_content"),
        ("Denied for card 4111111111111111 used on this device repeatedly.", "forbidden_content"),
        (
            "Denied for the cardholder on this device above the locked threshold.",
            "forbidden_content",
        ),
        (
            "Denied; the 250.00 USD attempt on this device exceeded the threshold.",
            "unsupported_number",
        ),
        (
            "This checkout was approved despite the locked threshold on this device.",
            "contradicts_decision",
        ),
    ],
)
def test_adversarial_output_is_rejected_and_never_returned(text: str, expected: str) -> None:
    with pytest.raises(ExplanationRejected, match=expected):
        verify(explanation(text), make_input())


def test_a_factor_the_input_does_not_support_is_rejected() -> None:
    """The load-bearing grounding check: a claim never given to the model."""

    payload = make_input(
        context=[
            EntityAggregate(
                kind="device",
                present=True,
                decision_count=4,
                denied_count=0,  # no prior denials
                review_count=4,
                span_seconds=201,
            )
        ]
    )
    response = LlmExplanation(
        summary="Denied because this device already carried prior denials in the window.",
        factors=[FactorCode.PRIOR_DENIALS_ON_DEVICE],
        caveat="",
    )
    with pytest.raises(ExplanationRejected, match="unsupported_factor"):
        verify(response, payload)


def test_duplicate_factors_are_rejected() -> None:
    response = LlmExplanation(
        summary="Denied because calibrated risk exceeded the locked threshold on this device.",
        factors=[
            FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD,
            FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD,
        ],
        caveat="",
    )
    with pytest.raises(ExplanationRejected, match="unsupported_factor"):
        verify(response, make_input())


def test_the_caveat_is_scanned_as_strictly_as_the_summary() -> None:
    with pytest.raises(ExplanationRejected, match="unsupported_number"):
        verify(
            explanation(
                "Denied because calibrated risk exceeded the locked threshold on this device.",
                caveat="Confidence is limited by only 9 observations.",
            ),
            make_input(),
        )


def test_normalise_makes_equal_values_compare_equal() -> None:
    assert normalise("250.00") == normalise("250") == "250"
    assert normalise("25,000") == "25000"
    assert normalise("0.0070") == "0.007"
    assert normalise("0.0070") != normalise(PROBABILITY)
