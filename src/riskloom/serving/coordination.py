"""Deterministic layout for the coordination graph.

Layout is computed here, in Python, so the signature view is covered by the same pytest gate as
everything else and so screenshots reproduce. The client receives finished coordinates and only
draws them.

The layout is presentation geometry only. It derives no probability, metric or feature value and
carries no risk semantics.

Determinism is a property of the algorithm, not a convention: there is no random source anywhere,
every ordering is derived from sorted node identifiers, and the one place a nudge is needed it is
hashed from the identifier with SHA-256. The same graph and the same canvas always produce the
same coordinates.
"""

import hashlib
import math
from dataclasses import dataclass, field

DEFAULT_CANVAS_WIDTH = 1_440
DEFAULT_CANVAS_HEIGHT = 700
MINIMUM_CANVAS_WIDTH = 480
MINIMUM_CANVAS_HEIGHT = 360
MAXIMUM_CANVAS_WIDTH = 4_096
MAXIMUM_CANVAS_HEIGHT = 4_096

LABEL_BAND = 18
"""Vertical room reserved beneath a hub for its label."""

EVENT_GAP = 6
"""Minimum clear space between two event dots."""

HUB_CLEARANCE = 16
"""Minimum clear space between an event dot and a hub's edge."""

EVENT_BOUND_PAD = 6
"""Event edge margin.

Deliberately separate from the hub margin. The hub margin exists to reserve label space beneath a
hub; applying it to events pinned them against the canvas edge where they could not slide far
enough to clear a neighbouring hub, which left a residual overlap on dense graphs.
"""

RELAX_PASSES = 320
"""Enough passes to separate the densest realistic graph.

The loop runs over events only, and a busy ledger window holds tens of nodes, not thousands.
"""

MINIMUM_ENTITY_RADIUS = 14
MAXIMUM_ENTITY_RADIUS = 40
EVENT_RADIUS = 7


@dataclass(frozen=True, slots=True)
class LayoutNode:
    """A node awaiting placement. Only geometry-relevant attributes appear here."""

    node_id: str
    is_hub: bool
    radius: int


@dataclass(frozen=True, slots=True)
class LayoutInput:
    nodes: list[LayoutNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    """Directed event -> hub pairs."""


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


class CoordinationLayoutError(ValueError):
    """A safe layout error."""


def clamp_canvas(width: int | None, height: int | None) -> tuple[int, int]:
    """Bound requested canvas dimensions to something drawable."""

    resolved_width = DEFAULT_CANVAS_WIDTH if width is None else width
    resolved_height = DEFAULT_CANVAS_HEIGHT if height is None else height
    return (
        max(MINIMUM_CANVAS_WIDTH, min(int(resolved_width), MAXIMUM_CANVAS_WIDTH)),
        max(MINIMUM_CANVAS_HEIGHT, min(int(resolved_height), MAXIMUM_CANVAS_HEIGHT)),
    )


def entity_radius(decision_count: int, shared_kinds: int) -> int:
    """Radius grows with attached decisions; the ring reads as 'how much reuse is here'."""

    scaled = MINIMUM_ENTITY_RADIUS + int(math.sqrt(max(decision_count, 1) - 1) * 9)
    scaled += min(shared_kinds, 3) * 2
    return min(scaled, MAXIMUM_ENTITY_RADIUS)


def label_offset(radius: int, index: int, hub_count: int) -> int:
    """Alternate label distance so neighbouring labels on a crowded ring do not collide."""

    base = radius + 13
    return base + 12 if hub_count > 8 and index % 2 == 1 else base


def _hash_unit(token: str) -> float:
    """A stable value in [0, 1) derived from an identifier. Never a random draw."""

    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return ((digest[0] << 8) | digest[1]) / 65_536.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def cluster_order(layout: LayoutInput) -> list[LayoutNode]:
    """Order hubs so that hubs sharing an event are neighbours on the ring.

    Without this, two hubs of the same cluster can land at opposite ring positions. Their shared
    events then average to the canvas centre, drift far from both hubs, and tangle with an
    unrelated cluster's events. Grouping by connected component keeps each cluster a contiguous,
    readable shape.

    Components are discovered from hubs in sorted order and neighbours are visited in sorted
    order, so the result is deterministic.
    """

    hubs = sorted((node for node in layout.nodes if node.is_hub), key=lambda node: node.node_id)
    hub_ids = {hub.node_id for hub in hubs}
    linked: dict[str, set[str]] = {hub.node_id: set() for hub in hubs}

    hubs_by_event: dict[str, list[str]] = {}
    for source, target in layout.edges:
        if target in hub_ids:
            hubs_by_event.setdefault(source, []).append(target)
    for group in hubs_by_event.values():
        for left in group:
            for right in group:
                if left != right:
                    linked[left].add(right)

    seen: set[str] = set()
    ordered: list[str] = []
    for hub in hubs:
        if hub.node_id in seen:
            continue
        queue = [hub.node_id]
        seen.add(hub.node_id)
        while queue:
            current = queue.pop(0)
            ordered.append(current)
            for neighbour in sorted(linked[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)

    by_id = {hub.node_id: hub for hub in hubs}
    return [by_id[node_id] for node_id in ordered]


def compute_layout(layout: LayoutInput, width: int, height: int) -> dict[str, Point]:
    """Place every node inside ``width`` x ``height``.

    Hubs sit on an ellipse scaled to the available box, ordered so a cluster is contiguous. Events
    are drawn toward the centroid of the hubs they actually connect to, so an event shared by two
    hubs lands between them and the shared structure is what the eye reads. A relaxation pass then
    separates anything overlapping.
    """

    if width < MINIMUM_CANVAS_WIDTH or height < MINIMUM_CANVAS_HEIGHT:
        raise CoordinationLayoutError("coordination_canvas_too_small")

    hubs = cluster_order(layout)
    events = sorted(
        (node for node in layout.nodes if not node.is_hub), key=lambda node: node.node_id
    )
    if not hubs and not events:
        return {}

    positions: dict[str, list[float]] = {}
    largest_hub = max((hub.radius for hub in hubs), default=0)
    margin = largest_hub + LABEL_BAND + 12
    centre_x, centre_y = width / 2, height / 2
    radius_x = max(40.0, width / 2 - margin)
    radius_y = max(40.0, height / 2 - margin)

    for index, hub in enumerate(hubs):
        if len(hubs) == 1:
            positions[hub.node_id] = [centre_x, centre_y]
            continue
        step = (2 * math.pi) / len(hubs)
        nudge = (_hash_unit(hub.node_id) - 0.5) * step * 0.28
        # Start along the wider axis. With only two hubs this is what puts them left and right on
        # a landscape canvas instead of stacking them into a narrow column.
        start = math.pi if radius_x >= radius_y else -math.pi / 2
        angle = start + index * step + nudge
        positions[hub.node_id] = [
            centre_x + math.cos(angle) * radius_x,
            centre_y + math.sin(angle) * radius_y,
        ]

    hubs_for_event: dict[str, list[str]] = {}
    for source, target in layout.edges:
        hubs_for_event.setdefault(source, []).append(target)

    links_signature = {
        event.node_id: "|".join(sorted(hubs_for_event.get(event.node_id, []))) for event in events
    }
    for index, event in enumerate(events):
        linked = [
            positions[hub] for hub in hubs_for_event.get(event.node_id, []) if hub in positions
        ]
        if linked:
            anchor_x = sum(point[0] for point in linked) / len(linked)
            anchor_y = sum(point[1] for point in linked) / len(linked)
        else:
            anchor_x, anchor_y = centre_x, centre_y

        siblings = [
            other.node_id
            for other in events
            if links_signature[other.node_id] == links_signature[event.node_id]
        ]
        rank = siblings.index(event.node_id)
        spread = 46.0 if len(linked) > 1 else float(largest_hub + HUB_CLEARANCE + 26)
        step = (2 * math.pi) / max(len(siblings), 1)
        angle = step * rank + _hash_unit(event.node_id) * 0.6 + index * 0.017
        positions[event.node_id] = [
            anchor_x + math.cos(angle) * spread,
            anchor_y + math.sin(angle) * spread,
        ]

    _relax(positions, hubs, events, width, height)

    return {
        node_id: Point(x=int(round(point[0])), y=int(round(point[1])))
        for node_id, point in positions.items()
    }


def _relax(
    positions: dict[str, list[float]],
    hubs: list[LayoutNode],
    events: list[LayoutNode],
    width: int,
    height: int,
) -> None:
    """Push overlapping events apart and keep everything inside the canvas.

    Hubs stay where the ellipse put them; only events move. That keeps hub spacing -- and
    therefore hub label spacing -- under the ring's control rather than at the mercy of relaxation.
    """

    hub_points = [(positions[hub.node_id], hub.radius) for hub in hubs]

    for _ in range(RELAX_PASSES):
        for event in events:
            point = positions[event.node_id]
            shift_x = 0.0
            shift_y = 0.0

            for other in events:
                if other.node_id == event.node_id:
                    continue
                target = positions[other.node_id]
                dx = point[0] - target[0]
                dy = point[1] - target[1]
                distance = math.hypot(dx, dy) or 0.001
                minimum = event.radius + other.radius + EVENT_GAP
                if distance < minimum:
                    push = (minimum - distance) / distance * 0.6
                    shift_x += dx * push
                    shift_y += dy * push

            for hub_point, hub_radius in hub_points:
                dx = point[0] - hub_point[0]
                dy = point[1] - hub_point[1]
                distance = math.hypot(dx, dy) or 0.001
                minimum = hub_radius + event.radius + HUB_CLEARANCE
                if distance < minimum:
                    push = (minimum - distance) / distance
                    shift_x += dx * push
                    shift_y += dy * push

            bound = event.radius + EVENT_BOUND_PAD
            point[0] = _clamp(point[0] + shift_x * 0.5, bound, width - bound)
            point[1] = _clamp(point[1] + shift_y * 0.5, bound, height - bound)

    # Final guarantee: nothing, including the label band beneath a hub, leaves the canvas.
    for hub in hubs:
        point = positions[hub.node_id]
        point[0] = _clamp(point[0], hub.radius + 4, width - hub.radius - 4)
        point[1] = _clamp(point[1], hub.radius + 4, height - hub.radius - LABEL_BAND - 4)
    for event in events:
        point = positions[event.node_id]
        point[0] = _clamp(point[0], event.radius + 4, width - event.radius - 4)
        point[1] = _clamp(point[1], event.radius + 4, height - event.radius - 4)
