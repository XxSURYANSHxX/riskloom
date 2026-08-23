"""Strict input and output schemas for explanation generation.

The input schema is the anti-injection design. Every field is a number, a bool, or a member of a
closed enum; not one free-text field is sent. A prompt-injection attack needs a channel for
attacker-controlled text to reach the model, and this contract has none -- there is no input a
caller can express that carries an instruction.

The output schema is the anti-hallucination design. ``factors`` is a closed enum rather than prose,
so an invented contributing factor is not filtered out, it is unrepresentable.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_SUMMARY_CHARS = 400
MAX_CAVEAT_CHARS = 300
MAX_FACTORS = 4


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FailSafeReasonInput(StrEnum):
    """Mirror of the serving enum.

    Duplicated deliberately rather than imported: this package must not import anything from
    ``riskloom.serving``, and an import purely for four constant strings would create exactly the
    coupling the isolation tests forbid. A test asserts the two enums stay identical.
    """

    FEATURE_COMPUTATION_FAILED = "feature_computation_failed"
    SCORING_FAILED = "scoring_failed"
    ORDER_CREATION_FAILED = "order_creation_failed"
    ORDER_BUDGET_EXHAUSTED = "order_budget_exhausted"


class FactorCode(StrEnum):
    """The closed vocabulary of contributing factors the model may select.

    RiskLoom renders the human sentence for each code from its own template and its own numbers.
    The model chooses codes; it never writes the factor text.
    """

    PROBABILITY_AT_OR_ABOVE_THRESHOLD = "probability_at_or_above_threshold"
    DEVICE_REUSE = "device_reuse"
    NETWORK_REUSE = "network_reuse"
    INSTRUMENT_REUSE = "instrument_reuse"
    MERCHANT_VOLUME = "merchant_volume"
    PRIOR_DENIALS_ON_DEVICE = "prior_denials_on_device"
    PRIOR_DENIALS_ON_NETWORK = "prior_denials_on_network"
    PRIOR_DENIALS_ON_INSTRUMENT = "prior_denials_on_instrument"
    RAPID_SUCCESSION = "rapid_succession"


EntityKind = Literal["device", "network", "instrument", "merchant"]


class EntityAggregate(_Strict):
    """Ledger co-occurrence for one entity, with the token deliberately absent.

    These are the same aggregates the Day 7 case-detail view already displays. The raw
    pseudonymous token is not carried here at all: the model has no legitimate use for it, and a
    field that does not exist cannot be sent by mistake.
    """

    kind: EntityKind
    present: bool
    decision_count: int = Field(ge=0)
    denied_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    span_seconds: int | None = Field(default=None, ge=0)


class ExplanationInput(_Strict):
    """The complete set of facts that may be sent to the model.

    Probabilities travel as strings at full stored precision. The locked threshold carries
    nineteen significant decimals and a real tie-cluster sits one unit in the last place below it,
    so a float round-trip could move a value across the boundary it is meant to describe.
    """

    calibrated_probability: str = Field(min_length=1, max_length=32)
    decision_threshold: str = Field(min_length=1, max_length=32)
    probability_exceeds_threshold: bool
    risk_decision: Literal["allow", "deny"]
    action: Literal["allow", "review", "deny"]
    fail_safe_reason: FailSafeReasonInput | None = None
    amount_subunits: int = Field(ge=1)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    channel: Literal["web", "mobile_web", "mobile_app"]
    context: list[EntityAggregate] = Field(default_factory=list, max_length=4)


class LlmExplanation(BaseModel):
    """The only shape a model response may take.

    Deliberately *not* ``strict=True``, unlike every other model here. Strict mode requires an
    actual ``FactorCode`` instance, and a model replies with JSON strings, so a strict version of
    this schema rejects every well-formed real response. The input models keep strict mode because
    RiskLoom constructs them itself with real typed values.

    Nothing is loosened by the change that matters: ``extra="forbid"`` still refuses an unknown
    key, an out-of-enum factor still fails, and Pydantic v2 does not coerce an int into a ``str``,
    so a numeric ``summary`` is still rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=20, max_length=MAX_SUMMARY_CHARS)
    factors: list[FactorCode] = Field(min_length=1, max_length=MAX_FACTORS)
    caveat: str = Field(default="", max_length=MAX_CAVEAT_CHARS)


class ExplanationRejected(ValueError):
    """A validated-shape response that failed a grounding or safety check.

    The message is always a short stable identity, never model output and never an upstream body.
    """
