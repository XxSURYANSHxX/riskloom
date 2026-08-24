"""Tests for the coordination layout.

These assert the geometry that actually reaches the screen. The client receives these coordinates
and draws them unchanged, so a regression here is a visible regression.
"""

import ast
import inspect
import math
from pathlib import Path

import pytest

from riskloom.serving import coordination
from riskloom.serving.coordination import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    EVENT_GAP,
    EVENT_RADIUS,
    HUB_CLEARANCE,
    LABEL_BAND,
    MAXIMUM_CANVAS_HEIGHT,
    MAXIMUM_CANVAS_WIDTH,
    MINIMUM_CANVAS_HEIGHT,
    MINIMUM_CANVAS_WIDTH,
    CoordinationLayoutError,
    LayoutInput,
    LayoutNode,
    clamp_canvas,
    cluster_order,
    compute_layout,
    entity_radius,
    label_offset,
)

CANVASES = [(1_100, 620), (1_440, 700), (820, 480)]


def hub(name: str, radius: int = 20) -> LayoutNode:
    return LayoutNode(node_id=name, is_hub=True, radius=radius)


def event(name: str) -> LayoutNode:
    return LayoutNode(node_id=name, is_hub=False, radius=EVENT_RADIUS)


def build(hub_count: int, events_per_hub: int, *, shared: bool = False) -> LayoutInput:
    """A synthetic graph. When ``shared``, each event links to two consecutive hubs."""

    hubs = [hub(f"device:{index:02d}", 18 + index % 6) for index in range(hub_count)]
    nodes: list[LayoutNode] = list(hubs)
    edges: list[tuple[str, str]] = []
    for hub_index in range(hub_count):
        for slot in range(events_per_hub):
            node_id = f"event:{hub_index:02d}-{slot}"
            nodes.append(event(node_id))
            edges.append((node_id, hubs[hub_index].node_id))
            if shared:
                edges.append((node_id, hubs[(hub_index + 1) % hub_count].node_id))
    return LayoutInput(nodes=nodes, edges=edges)


REAL_LEDGER = LayoutInput(
    nodes=[
        hub("device:ec57", 27),
        hub("network:ec57", 27),
        *(event(f"event:{index}") for index in range(4)),
    ],
    edges=[
        *((f"event:{index}", "device:ec57") for index in range(4)),
        *((f"event:{index}", "network:ec57") for index in range(4)),
    ],
)


def dense_pair(events: int) -> LayoutInput:
    """Two hubs, many events, every event on both.

    Orthogonal to every other fixture here. `build()` produces many hubs with few events each; this
    produces the opposite, which is exactly what repeated demo-burst clicks accumulate: one device
    hub and one network hub with the whole burst history hanging off both.
    """

    radius = entity_radius(events, 1)
    nodes = [
        hub("device:dense", radius),
        hub("network:dense", radius),
        *(event(f"event:{index:04d}") for index in range(events)),
    ]
    edges = [
        pair
        for index in range(events)
        for pair in (
            (f"event:{index:04d}", "device:dense"),
            (f"event:{index:04d}", "network:dense"),
        )
    ]
    return LayoutInput(nodes=nodes, edges=edges)


FIXTURES: list[tuple[str, LayoutInput]] = [
    ("real 2-hub burst", REAL_LEDGER),
    ("single hub", build(1, 5)),
    ("6 hubs", build(6, 1)),
    ("8 hubs shared", build(8, 2, shared=True)),
    ("15 hubs", build(15, 1)),
    ("20 hubs", build(20, 2)),
    ("20 hubs shared", build(20, 3, shared=True)),
    ("2 hubs, 40 events", dense_pair(40)),
    ("2 hubs, 60 events", dense_pair(60)),
]


def radius_of(layout: LayoutInput, node_id: str) -> int:
    return next(node.radius for node in layout.nodes if node.node_id == node_id)


# --------------------------------------------------------------------------- determinism


@pytest.mark.parametrize(("name", "layout"), FIXTURES, ids=[name for name, _ in FIXTURES])
@pytest.mark.parametrize(("width", "height"), CANVASES)
def test_layout_is_identical_across_repeated_calls(
    name: str, layout: LayoutInput, width: int, height: int
) -> None:
    first = compute_layout(layout, width, height)
    for _ in range(3):
        assert compute_layout(layout, width, height) == first


def test_layout_is_independent_of_node_and_edge_order() -> None:
    """Ordering comes from sorted identifiers, so shuffled input must land identically."""

    layout = build(8, 2, shared=True)
    shuffled = LayoutInput(
        nodes=list(reversed(layout.nodes)),
        edges=list(reversed(layout.edges)),
    )
    assert compute_layout(shuffled, 1_440, 700) == compute_layout(layout, 1_440, 700)


def test_layout_uses_no_random_source() -> None:
    """Any random draw would break reproducibility of a screenshot; there must be none."""

    tree = ast.parse(Path(inspect.getfile(coordination)).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not ({"random", "shuffle", "uniform", "randint", "sample", "time"} & names)
    assert "random" not in {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


# --------------------------------------------------------------------------- cluster ordering


def test_hubs_sharing_an_event_are_adjacent_in_the_ring_order() -> None:
    """The antipodal-hub fix.

    Two hubs of one cluster must be neighbours on the ring. Placed opposite each other, their
    shared events average to the canvas centre and tangle with an unrelated cluster.
    """

    layout = LayoutInput(
        nodes=[
            hub("device:a"),
            hub("device:b"),
            hub("device:c"),
            hub("device:d"),
            event("event:1"),
        ],
        # Sorted order interleaves the pair; only component ordering pulls a and c together.
        edges=[("event:1", "device:a"), ("event:1", "device:c")],
    )
    order = [node.node_id for node in cluster_order(layout)]
    assert abs(order.index("device:a") - order.index("device:c")) == 1


def test_a_shared_event_sits_with_its_own_cluster_not_an_unrelated_one() -> None:
    """The visible consequence of cluster ordering, asserted on coordinates.

    Deliberately not phrased as "nearer its hubs than the canvas centre": the centroid of two
    adjacent hubs on a four-hub ring is about as far from either hub as it is from the centre, so
    that comparison would pass or fail on ring size rather than on the behaviour under test. What
    must hold is that the event keeps company with the hubs it actually links to.
    """

    layout = LayoutInput(
        nodes=[
            hub("device:a"),
            hub("device:b"),
            hub("device:c"),
            hub("device:d"),
            event("event:1"),
        ],
        edges=[("event:1", "device:a"), ("event:1", "device:c")],
    )
    points = compute_layout(layout, 1_440, 700)
    spot = points["event:1"]

    def distance(name: str) -> float:
        return math.hypot(spot.x - points[name].x, spot.y - points[name].y)

    assert max(distance("device:a"), distance("device:c")) < min(
        distance("device:b"), distance("device:d")
    )


def test_cluster_order_keeps_every_hub_exactly_once() -> None:
    layout = build(15, 1, shared=True)
    ordered = [node.node_id for node in cluster_order(layout)]
    assert sorted(ordered) == sorted(node.node_id for node in layout.nodes if node.is_hub)
    assert len(ordered) == len(set(ordered))


def test_disconnected_hubs_are_still_all_placed() -> None:
    layout = LayoutInput(nodes=[hub("device:a"), hub("device:b")], edges=[])
    assert set(compute_layout(layout, 1_440, 700)) == {"device:a", "device:b"}


# --------------------------------------------------------------------------- placement quality


@pytest.mark.parametrize(("name", "layout"), FIXTURES, ids=[name for name, _ in FIXTURES])
@pytest.mark.parametrize(("width", "height"), CANVASES)
def test_nothing_is_clipped_by_the_canvas(
    name: str, layout: LayoutInput, width: int, height: int
) -> None:
    points = compute_layout(layout, width, height)
    for node in layout.nodes:
        point = points[node.node_id]
        reserved = LABEL_BAND if node.is_hub else 0
        assert node.radius <= point.x <= width - node.radius, f"{name}: {node.node_id} x"
        assert node.radius <= point.y <= height - node.radius - reserved, (
            f"{name}: {node.node_id} y"
        )


@pytest.mark.parametrize(("name", "layout"), FIXTURES, ids=[name for name, _ in FIXTURES])
@pytest.mark.parametrize(("width", "height"), CANVASES)
def test_no_two_nodes_overlap(name: str, layout: LayoutInput, width: int, height: int) -> None:
    points = compute_layout(layout, width, height)
    nodes = sorted(layout.nodes, key=lambda node: node.node_id)
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            gap = math.hypot(
                points[left.node_id].x - points[right.node_id].x,
                points[left.node_id].y - points[right.node_id].y,
            )
            assert gap > left.radius + right.radius, f"{name}: {left.node_id}/{right.node_id}"


@pytest.mark.parametrize(("name", "layout"), FIXTURES, ids=[name for name, _ in FIXTURES])
def test_hub_labels_do_not_collide(name: str, layout: LayoutInput) -> None:
    """Labels sit beneath their hub; two hub label anchors must not land on the same spot."""

    points = compute_layout(layout, 1_440, 700)
    hubs = cluster_order(layout)
    anchors = [
        (
            points[node.node_id].x,
            points[node.node_id].y + label_offset(node.radius, index, len(hubs)),
        )
        for index, node in enumerate(hubs)
    ]
    for index, left in enumerate(anchors):
        for right in anchors[index + 1 :]:
            assert math.hypot(left[0] - right[0], left[1] - right[1]) > 24, name


def test_events_may_sit_closer_to_the_edge_than_a_hub_margin_allows() -> None:
    """The event-clamping fix.

    Events are bounded by their own small pad, not by the hub margin. Clamped to the hub margin,
    an event could not slide far enough to clear a neighbouring hub and one overlap survived on
    dense graphs. This asserts the bound is genuinely tighter than the hub margin.
    """

    layout = build(20, 3, shared=True)
    points = compute_layout(layout, 1_100, 620)
    hub_margin = max(node.radius for node in layout.nodes if node.is_hub) + LABEL_BAND + 12
    closest = min(
        min(points[node.node_id].x, 1_100 - points[node.node_id].x)
        for node in layout.nodes
        if not node.is_hub
    )
    assert closest < hub_margin


def test_events_are_drawn_toward_the_hubs_they_link_to() -> None:
    """Centroid placement: an event's own hub must be nearer than an unrelated one."""

    layout = build(6, 1)
    points = compute_layout(layout, 1_440, 700)
    spot = points["event:00-0"]
    own = math.hypot(spot.x - points["device:00"].x, spot.y - points["device:00"].y)
    others = [
        math.hypot(
            spot.x - points[f"device:{index:02d}"].x, spot.y - points[f"device:{index:02d}"].y
        )
        for index in range(1, 6)
    ]
    assert own < min(others)


def test_two_hubs_are_separated_along_the_wider_axis() -> None:
    """On a landscape canvas a pair must read left-and-right, not stacked into a column."""

    layout = LayoutInput(nodes=[hub("device:a"), hub("device:b")], edges=[])
    points = compute_layout(layout, 1_440, 700)
    horizontal = abs(points["device:a"].x - points["device:b"].x)
    vertical = abs(points["device:a"].y - points["device:b"].y)
    assert horizontal > vertical


def test_events_clear_their_hub_and_each_other() -> None:
    layout = build(4, 4)
    points = compute_layout(layout, 1_440, 700)
    events = [node for node in layout.nodes if not node.is_hub]
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            gap = math.hypot(
                points[left.node_id].x - points[right.node_id].x,
                points[left.node_id].y - points[right.node_id].y,
            )
            assert gap >= left.radius + right.radius + EVENT_GAP - 1
    for node in events:
        for hub_node in (item for item in layout.nodes if item.is_hub):
            gap = math.hypot(
                points[node.node_id].x - points[hub_node.node_id].x,
                points[node.node_id].y - points[hub_node.node_id].y,
            )
            assert gap >= hub_node.radius + node.radius + HUB_CLEARANCE - 1


def test_the_graph_uses_a_real_share_of_the_canvas() -> None:
    """The fill regression that started this work: events once occupied 24% x 30% of the box."""

    layout = build(20, 2)
    for width, height in CANVASES:
        points = compute_layout(layout, width, height)
        xs = [point.x for point in points.values()]
        ys = [point.y for point in points.values()]
        assert (max(xs) - min(xs)) / width > 0.7
        assert (max(ys) - min(ys)) / height > 0.7


def test_a_single_hub_sits_at_the_canvas_centre() -> None:
    points = compute_layout(LayoutInput(nodes=[hub("device:a")], edges=[]), 1_440, 700)
    assert (points["device:a"].x, points["device:a"].y) == (720, 350)


def test_an_empty_graph_produces_no_coordinates() -> None:
    assert compute_layout(LayoutInput(), 1_440, 700) == {}


# --------------------------------------------------------------------------- canvas handling


def test_the_same_graph_moves_when_the_canvas_changes() -> None:
    """Coordinates are canvas-relative; identical output would mean the size was ignored."""

    layout = build(6, 1)
    assert compute_layout(layout, 1_440, 700) != compute_layout(layout, 820, 480)


def test_canvas_dimensions_are_clamped_to_a_drawable_range() -> None:
    assert clamp_canvas(None, None) == (DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)
    assert clamp_canvas(0, 0) == (MINIMUM_CANVAS_WIDTH, MINIMUM_CANVAS_HEIGHT)
    assert clamp_canvas(-40, -40) == (MINIMUM_CANVAS_WIDTH, MINIMUM_CANVAS_HEIGHT)
    assert clamp_canvas(99_999, 99_999) == (MAXIMUM_CANVAS_WIDTH, MAXIMUM_CANVAS_HEIGHT)
    assert clamp_canvas(1_100, 620) == (1_100, 620)


def test_an_undrawably_small_canvas_is_refused() -> None:
    with pytest.raises(CoordinationLayoutError, match="coordination_canvas_too_small"):
        compute_layout(build(4, 1), 100, 100)


# --------------------------------------------------------------------------- sizing


def test_radius_grows_with_reuse_and_is_bounded() -> None:
    assert entity_radius(1, 0) < entity_radius(4, 0) < entity_radius(40, 0)
    assert entity_radius(2, 2) > entity_radius(2, 0)
    assert entity_radius(10_000, 3) <= coordination.MAXIMUM_ENTITY_RADIUS
    assert entity_radius(0, 0) >= coordination.MINIMUM_ENTITY_RADIUS


def test_label_offset_alternates_only_on_a_crowded_ring() -> None:
    assert label_offset(20, 0, 4) == label_offset(20, 1, 4)
    assert label_offset(20, 1, 12) > label_offset(20, 0, 12)
    assert label_offset(30, 0, 4) > label_offset(20, 0, 4)


# --------------------------------------------------------------------------- dense pair


def test_a_dense_hub_pair_stays_separated_at_burst_scale() -> None:
    """Two hubs carrying a whole burst history: no overlap, at every canvas size.

    This shape was an untested gap. Every other fixture spreads events across many hubs; repeated
    demo-burst clicks do the opposite, piling 40-60 events onto one device hub and one network hub.
    The layout copes -- relaxation keeps every pair apart -- and that is what is locked in here.
    """

    for events in (40, 60):
        layout = dense_pair(events)
        for width, height in CANVASES:
            points = compute_layout(layout, width, height)
            nodes = sorted(layout.nodes, key=lambda node: node.node_id)
            for index, left in enumerate(nodes):
                for right in nodes[index + 1 :]:
                    gap = math.hypot(
                        points[left.node_id].x - points[right.node_id].x,
                        points[left.node_id].y - points[right.node_id].y,
                    )
                    assert gap > left.radius + right.radius, (events, width, left.node_id)


def test_the_dense_pair_is_crowded_and_that_is_recorded_not_hidden() -> None:
    """The honest part: the graph is dense at this scale, and stays legible only just.

    At 60 events on one hub pair, most event dots sit within about 20px of a neighbour -- clear of
    each other, but reading as a cluster rather than as countable points. This asserts the measured
    state so it is a known, intentional limitation rather than a surprise in a demo. Collapsing
    events into meta-nodes would fix the density and is deliberately *not* done here; see the
    deferred-improvement note in the report.
    """

    layout = dense_pair(60)
    points = compute_layout(layout, 1_100, 620)
    events = [node for node in layout.nodes if not node.is_hub]

    gaps = []
    for left in events:
        nearest = min(
            math.hypot(
                points[left.node_id].x - points[right.node_id].x,
                points[left.node_id].y - points[right.node_id].y,
            )
            for right in events
            if right.node_id != left.node_id
        )
        gaps.append(nearest)

    # Never touching: the separation guarantee holds even at this density.
    assert min(gaps) > 2 * EVENT_RADIUS
    # But crowded: most dots have a close neighbour. Recorded, not asserted away.
    crowded = sum(1 for gap in gaps if gap < 20)
    assert crowded > len(events) // 2, crowded


def test_the_dense_pair_still_uses_the_canvas_horizontally() -> None:
    """Density must not collapse the graph into a corner, which was the Day 7 regression."""

    points = compute_layout(dense_pair(60), 1_100, 620)
    xs = [point.x for point in points.values()]
    assert (max(xs) - min(xs)) / 1_100 > 0.7
