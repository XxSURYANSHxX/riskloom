"""Read-only ledger queries backing the dashboard.

Everything here is a projection of rows already written by the live scoring path. No probability,
metric or feature value is derived: figures are read from the column where the decision that
produced them stored them.

Coordination edges come from grouping stored token columns. That is a SQL projection of persisted
data, not campaign detection — the live path emits no campaign, and this module must never imply
that it does.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.dashboard.schemas import (
    ActionCounts,
    CoordinationGraph,
    DecisionDetail,
    DecisionPage,
    DecisionSummary,
    EntityContext,
    GraphEdge,
    GraphNode,
    LedgerSummary,
)
from riskloom.db.models import ReviewItem, RiskDecision
from riskloom.serving.coordination import (
    EVENT_RADIUS,
    LayoutInput,
    LayoutNode,
    clamp_canvas,
    compute_layout,
    entity_radius,
    label_offset,
)

MAXIMUM_PAGE = 500
DEFAULT_WINDOW_SECONDS = 3_600

# The entity kinds whose reuse defines coordination. Merchant is deliberately excluded from the
# graph: sharing a merchant is ordinary (every checkout has one) and would make a hub of the whole
# canvas. It remains available as context on the case-detail view.
GRAPH_ENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("device", "device_token"),
    ("network", "network_token"),
    ("instrument", "payment_instrument_token"),
)
CONTEXT_ENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    *GRAPH_ENTITY_FIELDS,
    ("merchant", "merchant_id"),
)


@dataclass(frozen=True, slots=True)
class DecisionFilter:
    limit: int = 50
    offset: int = 0
    action: str | None = None
    since: datetime | None = None


def _probability_text(value: Decimal | None) -> str | None:
    """Render an exact decimal without float round-tripping."""

    return None if value is None else format(value, "f")


def to_summary(row: RiskDecision) -> DecisionSummary:
    return DecisionSummary(
        decision_id=row.id,
        event_id=row.event_id,
        merchant_id=row.merchant_id,
        checkout_id=row.checkout_id,
        device_token=row.device_token,
        network_token=row.network_token,
        payment_instrument_token=row.payment_instrument_token,
        session_token=row.session_token,
        customer_token=row.customer_token,
        amount_subunits=row.amount_subunits,
        currency=row.currency,
        channel=row.channel,
        occurred_at=row.occurred_at,
        created_at=row.created_at,
        calibrated_probability=_probability_text(row.calibrated_probability),
        decision_threshold=_probability_text(row.decision_threshold) or "0",
        risk_decision=row.risk_decision,
        action=row.action or "review",
        fail_safe_reason=row.fail_safe_reason,
        razorpay_order_id=row.razorpay_order_id,
        status=row.status,
        model_id=row.model_id,
    )


def _ordered() -> Select[tuple[RiskDecision]]:
    return select(RiskDecision).order_by(
        func.coalesce(RiskDecision.occurred_at, RiskDecision.created_at).desc(),
        RiskDecision.created_at.desc(),
    )


async def list_decisions(session: AsyncSession, filters: DecisionFilter) -> DecisionPage:
    limit = max(1, min(filters.limit, MAXIMUM_PAGE))
    offset = max(0, filters.offset)

    statement = _ordered()
    counter = select(func.count()).select_from(RiskDecision)
    if filters.action:
        statement = statement.where(RiskDecision.action == filters.action)
        counter = counter.where(RiskDecision.action == filters.action)
    if filters.since:
        anchor = func.coalesce(RiskDecision.occurred_at, RiskDecision.created_at)
        statement = statement.where(anchor >= filters.since)
        counter = counter.where(anchor >= filters.since)

    total = int(await session.scalar(counter) or 0)
    rows = list(await session.scalars(statement.limit(limit).offset(offset)))
    return DecisionPage(
        decisions=[to_summary(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


async def get_decision(session: AsyncSession, decision_id: str) -> DecisionDetail | None:
    row = await session.scalar(select(RiskDecision).where(RiskDecision.id == decision_id))
    if row is None:
        return None
    pending = await session.scalar(
        select(func.count()).select_from(ReviewItem).where(ReviewItem.risk_decision_id == row.id)
    )
    return DecisionDetail(
        decision=to_summary(row),
        context=await _entity_context(session, row),
        review_pending=bool(pending),
    )


async def _entity_context(session: AsyncSession, row: RiskDecision) -> list[EntityContext]:
    """Ledger co-occurrence for each token on this decision.

    This answers "how often has this token been seen, and how did those decisions go" purely from
    stored rows. It is not the model's feature vector, which is not persisted anywhere.
    """

    contexts: list[EntityContext] = []
    for kind, field in CONTEXT_ENTITY_FIELDS:
        token = getattr(row, field)
        if token is None:
            contexts.append(
                EntityContext(
                    kind=kind,  # type: ignore[arg-type]
                    token=None,
                    decision_count=0,
                    denied_count=0,
                    review_count=0,
                    first_seen_at=None,
                    last_seen_at=None,
                    span_seconds=None,
                )
            )
            continue
        column = getattr(RiskDecision, field)
        anchor = func.coalesce(RiskDecision.occurred_at, RiskDecision.created_at)
        result = (
            await session.execute(
                select(
                    func.count(),
                    func.count().filter(RiskDecision.action == "deny"),
                    func.count().filter(RiskDecision.action == "review"),
                    func.min(anchor),
                    func.max(anchor),
                ).where(column == token)
            )
        ).one()
        total, denied, reviewed, first_seen, last_seen = result
        span = (
            int((last_seen - first_seen).total_seconds())
            if first_seen is not None and last_seen is not None
            else None
        )
        contexts.append(
            EntityContext(
                kind=kind,  # type: ignore[arg-type]
                token=token,
                decision_count=int(total or 0),
                denied_count=int(denied or 0),
                review_count=int(reviewed or 0),
                first_seen_at=first_seen,
                last_seen_at=last_seen,
                span_seconds=span,
            )
        )
    return contexts


async def summarise(session: AsyncSession) -> LedgerSummary:
    counts = (
        await session.execute(
            select(
                func.count(),
                func.count().filter(RiskDecision.action == "allow"),
                func.count().filter(RiskDecision.action == "review"),
                func.count().filter(RiskDecision.action == "deny"),
                func.count().filter(RiskDecision.razorpay_order_id.is_not(None)),
                func.max(func.coalesce(RiskDecision.occurred_at, RiskDecision.created_at)),
            ).select_from(RiskDecision)
        )
    ).one()
    total, allow, review, deny, orders, latest = counts
    pending = await session.scalar(
        select(func.count()).select_from(ReviewItem).where(ReviewItem.status == "pending")
    )
    newest = await session.scalar(_ordered().limit(1))
    return LedgerSummary(
        total_decisions=int(total or 0),
        actions=ActionCounts(allow=int(allow or 0), review=int(review or 0), deny=int(deny or 0)),
        review_items_pending=int(pending or 0),
        orders_created=int(orders or 0),
        latest_decision_at=latest,
        model_id=newest.model_id if newest else None,
        feature_schema_version=newest.feature_schema_version if newest else None,
        feature_engine_version=newest.feature_engine_version if newest else None,
    )


async def build_coordination_graph(
    session: AsyncSession,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> CoordinationGraph:
    """Group stored tokens into a shared-token graph and lay it out.

    Only entities reused by more than one decision become hubs: a token seen once carries no
    coordination signal and would only add noise to the canvas.

    Topology is built first and laid out second. Keeping the two apart is what lets an event
    shared by several hubs sit between all of them: the layout sees the event's complete edge set,
    where a single pass that positioned each event as its hub was emitted could only ever anchor it
    to whichever hub happened to come first.
    """

    window = max(60, min(window_seconds, 86_400))
    width, height = clamp_canvas(canvas_width, canvas_height)
    anchor = func.coalesce(RiskDecision.occurred_at, RiskDecision.created_at)
    since = datetime.now(UTC) - timedelta(seconds=window)
    rows = list(await session.scalars(_ordered().where(anchor >= since)))

    shared: dict[str, tuple[str, str, list[RiskDecision]]] = {}
    for kind, field in GRAPH_ENTITY_FIELDS:
        grouped: dict[str, list[RiskDecision]] = {}
        for row in rows:
            token = getattr(row, field)
            if token:
                grouped.setdefault(token, []).append(row)
        for token, attached in grouped.items():
            if len(attached) > 1:
                shared[f"{kind}:{token}"] = (kind, token, attached)

    # How many distinct entity kinds each decision participates in; drives ring thickness.
    kinds_per_event: dict[str, set[str]] = {}
    for kind, _token, attached in shared.values():
        for row in attached:
            kinds_per_event.setdefault(str(row.id), set()).add(kind)

    # Phase one: topology and sizes, with no coordinates yet.
    hub_specs: list[tuple[str, str, str, int, int]] = []
    event_rows: dict[str, RiskDecision] = {}
    edges: list[GraphEdge] = []

    for node_id in sorted(shared):
        kind, token, attached = shared[node_id]
        # How many distinct entity kinds this hub's decisions also cluster on. A device whose
        # traffic additionally shares a network reads as more coordinated than one that does not.
        shared_kinds = len(
            {
                other_kind
                for row in attached
                for other_kind in kinds_per_event.get(str(row.id), set())
                if other_kind != kind
            }
        )
        hub_specs.append(
            (node_id, kind, token, entity_radius(len(attached), shared_kinds), shared_kinds)
        )
        for row in attached:
            event_node_id = f"event:{row.id}"
            event_rows.setdefault(event_node_id, row)
            edges.append(GraphEdge(source=event_node_id, target=node_id, weight=1))

    # Phase two: one layout pass over the complete graph.
    layout = LayoutInput(
        nodes=[
            *(
                LayoutNode(node_id=node_id, is_hub=True, radius=radius)
                for node_id, _kind, _token, radius, _shared in hub_specs
            ),
            *(
                LayoutNode(node_id=node_id, is_hub=False, radius=EVENT_RADIUS)
                for node_id in sorted(event_rows)
            ),
        ],
        edges=[(edge.source, edge.target) for edge in edges],
    )
    positions = compute_layout(layout, width, height)

    nodes: list[GraphNode] = []
    for index, (node_id, kind, token, radius, shared_kinds) in enumerate(hub_specs):
        point = positions[node_id]
        attached = shared[node_id][2]
        nodes.append(
            GraphNode(
                node_id=node_id,
                kind=kind,  # type: ignore[arg-type]
                label=token,
                x=point.x,
                y=point.y,
                radius=radius,
                label_offset=label_offset(radius, index, len(hub_specs)),
                action=None,
                decision_id=None,
                degree=len(attached),
                shared_kinds=shared_kinds,
            )
        )

    for event_node_id in sorted(event_rows):
        row = event_rows[event_node_id]
        point = positions[event_node_id]
        kinds = len(kinds_per_event.get(str(row.id), set()))
        nodes.append(
            GraphNode(
                node_id=event_node_id,
                kind="event",
                label=row.event_id,
                x=point.x,
                y=point.y,
                radius=EVENT_RADIUS,
                label_offset=EVENT_RADIUS + 13,
                action=row.action or "review",  # type: ignore[arg-type]
                decision_id=row.id,
                degree=kinds,
                shared_kinds=kinds,
            )
        )

    return CoordinationGraph(
        nodes=nodes,
        edges=sorted(edges, key=lambda edge: (edge.target, edge.source)),
        window_seconds=window,
        canvas_width=width,
        canvas_height=height,
        decision_count=len(rows),
        clustered_entity_count=len(shared),
    )
