"""Factor codes: their support predicates and their human renderings.

Every code the model selects must be *entailed by the input*. A model that reports prior denials on
a device whose ``denied_count`` is zero is not making a stylistic error, it is stating a fact that
was never given to it, and the response is rejected.

The rendered sentence is built here, from RiskLoom's own template and RiskLoom's own numbers. The
model contributes the selection and nothing else.
"""

from collections.abc import Callable

from riskloom.explanations.schemas import EntityAggregate, ExplanationInput, FactorCode

RAPID_SUCCESSION_MIN_DECISIONS = 3
RAPID_SUCCESSION_MAX_SPAN_SECONDS = 300
MERCHANT_VOLUME_MIN_DECISIONS = 5


def _entity(payload: ExplanationInput, kind: str) -> EntityAggregate | None:
    for aggregate in payload.context:
        if aggregate.kind == kind and aggregate.present:
            return aggregate
    return None


def _reused(payload: ExplanationInput, kind: str) -> bool:
    aggregate = _entity(payload, kind)
    return aggregate is not None and aggregate.decision_count > 1


def _denied(payload: ExplanationInput, kind: str) -> bool:
    aggregate = _entity(payload, kind)
    return aggregate is not None and aggregate.denied_count > 0


def _merchant_volume(payload: ExplanationInput) -> bool:
    aggregate = _entity(payload, "merchant")
    return aggregate is not None and aggregate.decision_count >= MERCHANT_VOLUME_MIN_DECISIONS


def _rapid_succession(payload: ExplanationInput) -> bool:
    for aggregate in payload.context:
        if not aggregate.present or aggregate.span_seconds is None:
            continue
        if (
            aggregate.decision_count >= RAPID_SUCCESSION_MIN_DECISIONS
            and aggregate.span_seconds <= RAPID_SUCCESSION_MAX_SPAN_SECONDS
        ):
            return True
    return False


SUPPORT: dict[FactorCode, Callable[[ExplanationInput], bool]] = {
    FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD: lambda p: p.probability_exceeds_threshold,
    FactorCode.DEVICE_REUSE: lambda p: _reused(p, "device"),
    FactorCode.NETWORK_REUSE: lambda p: _reused(p, "network"),
    FactorCode.INSTRUMENT_REUSE: lambda p: _reused(p, "instrument"),
    FactorCode.MERCHANT_VOLUME: _merchant_volume,
    FactorCode.PRIOR_DENIALS_ON_DEVICE: lambda p: _denied(p, "device"),
    FactorCode.PRIOR_DENIALS_ON_NETWORK: lambda p: _denied(p, "network"),
    FactorCode.PRIOR_DENIALS_ON_INSTRUMENT: lambda p: _denied(p, "instrument"),
    FactorCode.RAPID_SUCCESSION: _rapid_succession,
}

_LABELS: dict[FactorCode, str] = {
    FactorCode.PROBABILITY_AT_OR_ABOVE_THRESHOLD: "risk at or above the locked threshold",
    FactorCode.DEVICE_REUSE: "device reused across decisions",
    FactorCode.NETWORK_REUSE: "network reused across decisions",
    FactorCode.INSTRUMENT_REUSE: "instrument reused across decisions",
    FactorCode.MERCHANT_VOLUME: "elevated merchant volume in window",
    FactorCode.PRIOR_DENIALS_ON_DEVICE: "prior denials on this device",
    FactorCode.PRIOR_DENIALS_ON_NETWORK: "prior denials on this network",
    FactorCode.PRIOR_DENIALS_ON_INSTRUMENT: "prior denials on this instrument",
    FactorCode.RAPID_SUCCESSION: "decisions in rapid succession",
}


def is_supported(code: FactorCode, payload: ExplanationInput) -> bool:
    """Whether the input entails this factor."""

    return SUPPORT[code](payload)


def unsupported(codes: list[FactorCode], payload: ExplanationInput) -> list[FactorCode]:
    """Every selected code the input does not entail, in selection order."""

    return [code for code in codes if not is_supported(code, payload)]


def render(code: FactorCode) -> str:
    """RiskLoom's own wording for a factor. Never model text."""

    return _LABELS[code]


def available(payload: ExplanationInput) -> list[FactorCode]:
    """The codes this input actually supports.

    The prompt offers the model only these, so an unsupported selection means the model ignored an
    explicit constraint rather than merely guessing from a long menu.
    """

    return [code for code in FactorCode if is_supported(code, payload)]
