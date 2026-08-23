"""Dashboard response models.

These models are the PII allowlist. Every dashboard response is an explicit projection built from
named fields, never a passthrough of a database row or an artifact file, so a field can only reach
a client if it is written here deliberately. Tests assert the field sets match exactly.

Probabilities and thresholds serialise as **strings**, not floats. The locked threshold carries
nineteen significant decimals and Gate C1 established that a real tie-cluster of scored rows sits
one unit in the last place below it, so a JSON float round-trip could move a value across the
boundary it is meant to describe.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

DecisionAction = Literal["allow", "review", "deny"]
EntityKind = Literal["device", "network", "instrument", "merchant"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionSummary(BaseModel):
    """One ledger row as shown in the stream and the ledger table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: UUID
    event_id: str
    merchant_id: str
    checkout_id: str
    device_token: str | None
    network_token: str | None
    payment_instrument_token: str
    session_token: str
    customer_token: str | None
    amount_subunits: int
    currency: str
    channel: str
    occurred_at: datetime | None
    created_at: datetime
    calibrated_probability: str | None
    decision_threshold: str
    risk_decision: str | None
    action: str
    fail_safe_reason: str | None
    razorpay_order_id: str | None
    status: str
    model_id: str


class DecisionPage(_Model):
    decisions: list[DecisionSummary]
    total: int
    limit: int
    offset: int


class EntityContext(_Model):
    """Ledger co-occurrence for one entity token.

    This is a projection of stored tokens, not a model feature. It is deliberately never labelled
    as one: the 75-feature vector is not persisted, and recomputing it would produce values the
    decision never saw.
    """

    kind: EntityKind
    token: str | None
    decision_count: int
    denied_count: int
    review_count: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    span_seconds: int | None


class DecisionDetail(_Model):
    decision: DecisionSummary
    context: list[EntityContext]
    review_pending: bool


class ExplanationView(_Model):
    """A generated explanation as the dashboard sees it.

    ``factors`` carries RiskLoom's own rendering of each selected code, never model text; the codes
    themselves travel alongside so the client can style them without re-deriving anything. The
    model's prose reaches a client only through ``summary`` and ``caveat``, both of which have
    passed the grounding and forbidden-content checks before storage.
    """

    status: Literal["pending", "ready", "failed", "rejected"]
    summary: str | None
    factors: list[str]
    factor_codes: list[str]
    caveat: str | None
    failure_reason: str | None
    model_name: str
    prompt_version: str
    attempt_number: int
    attempts_remaining: int
    created_at: datetime


class ActionCounts(_Model):
    allow: int
    review: int
    deny: int


class LedgerSummary(_Model):
    total_decisions: int
    actions: ActionCounts
    review_items_pending: int
    orders_created: int
    latest_decision_at: datetime | None
    model_id: str | None
    feature_schema_version: str | None
    feature_engine_version: str | None


class GraphNode(_Model):
    node_id: str
    kind: Literal["event", "device", "network", "instrument"]
    label: str
    x: int
    y: int
    radius: int
    label_offset: int
    action: DecisionAction | None
    decision_id: UUID | None
    degree: int
    shared_kinds: int


class GraphEdge(_Model):
    source: str
    target: str
    weight: int


class CoordinationGraph(_Model):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    window_seconds: int
    canvas_width: int
    canvas_height: int
    decision_count: int
    clustered_entity_count: int


# --------------------------------------------------------------------------------------------
# Offline evaluation projection.
#
# evaluation.json is aggregate-only by construction (Gate B2 verified it contains no event ids,
# no campaign ids and no per-event prediction array), but this projection never passes the file
# through. Each field below is selected by name, so even if a future evaluation artifact gained a
# per-event section it could not reach a client.
# --------------------------------------------------------------------------------------------


class EvaluationThreshold(_Model):
    row_count: int
    attack_count: int
    legitimate_count: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float | None
    recall: float | None
    specificity: float | None
    f1_score: float | None
    false_positive_rate: float | None
    false_positives_per_10000_legitimate: float | None
    prevalence: float | None
    review_count: int
    review_rate: float | None
    cost_units: int
    cost_per_10000_events: float | None
    threshold: str


class EvaluationProbability(_Model):
    average_precision: float
    pr_auc_trapezoidal: float
    roc_auc: float
    brier_score: float
    log_loss: float


class ReliabilityBin(_Model):
    lower_inclusive: float
    upper_value: float
    upper_inclusive: bool
    count: int
    mean_probability: float | None
    attack_rate: float | None


class EvaluationReliability(_Model):
    expected_calibration_error: float
    bins: list[ReliabilityBin]


class HardNegativeSlice(_Model):
    slice_name: str
    row_count: int
    false_positive_count: int
    false_positive_rate: float | None
    false_positives_per_10000: float | None


class EvaluationCampaigns(_Model):
    campaign_count: int
    detected_campaign_count: int
    missed_campaign_count: int
    campaign_recall: float | None
    detection_delay_ms_minimum: int | None
    detection_delay_ms_median: int | None
    detection_delay_ms_p95: int | None
    detection_delay_ms_maximum: int | None
    flagged_per_campaign_minimum: int | None
    flagged_per_campaign_median: int | None
    flagged_per_campaign_maximum: int | None


class ModelEvaluation(_Model):
    """Offline held-out evaluation aggregates. Never live, and labelled as such in the UI."""

    model_id: str
    evaluation_id: str
    row_count: int
    threshold: EvaluationThreshold
    probability: EvaluationProbability
    reliability: EvaluationReliability
    hard_negative_slices: list[HardNegativeSlice]
    campaigns: EvaluationCampaigns
