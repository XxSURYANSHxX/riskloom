// Coordination graph renderer.
//
// Everything geometric -- positions, radii, label offsets -- is computed server-side in Python and
// arrives finished. This module only draws. The one number it contributes is the measured panel
// size, which it reads from the DOM and forwards as a query parameter, because the server has no
// way to know how wide the viewport is.

import { shortToken } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const MIN_HEIGHT = 420;

function node(name, attrs) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

/**
 * Read the canvas size to request a layout for.
 *
 * Bounded to the same range the endpoint accepts so a hidden or not-yet-laid-out panel (which
 * measures zero) asks for a sane default instead of a rejected one.
 */
export function measure(svg) {
  const box = svg ? svg.getBoundingClientRect() : { width: 0, height: 0 };
  const width = Math.max(Math.min(Math.round(box.width) || 960, 4096), 480);
  const height = Math.max(Math.min(Math.round(box.height) || MIN_HEIGHT, 4096), MIN_HEIGHT);
  return { width, height };
}

/**
 * Draw the graph.
 *
 * The visual argument of this screen: individually unremarkable events become one shape when they
 * share an identity. Hubs are entities, not events, so reuse is what grows on the canvas.
 */
export function renderGraph(svg, graph, { onSelect, previousIds } = {}) {
  // The viewBox tracks the canvas the server laid out against, so one SVG unit is one CSS pixel
  // and stroke widths and label sizes stay honest at any panel width.
  const width = graph.canvas_width;
  const height = graph.canvas_height;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();

  if (!graph.nodes.length) {
    const message = node("text", {
      x: width / 2, y: height / 2, "text-anchor": "middle", class: "hub-label", "font-size": 12,
    });
    message.textContent =
      "No shared tokens in this window. Run a burst from the stream view to form one.";
    svg.appendChild(message);
    return;
  }

  const byId = new Map(graph.nodes.map((item) => [item.node_id, item]));
  const at = (id) => byId.get(id);
  const seen = previousIds || new Set();
  const hubs = graph.nodes.filter((n) => n.kind !== "event");

  const edgeLayer = node("g", {});
  for (const edge of graph.edges) {
    const source = at(edge.source);
    const target = at(edge.target);
    if (!source || !target) continue;
    // A gentle bow perpendicular to the edge keeps parallel links into the same hub separable.
    const [dx, dy] = [target.x - source.x, target.y - source.y];
    const length = Math.hypot(dx, dy) || 1;
    const bow = Math.min(length * 0.12, 26);
    const midX = (source.x + target.x) / 2 - (dy / length) * bow;
    const midY = (source.y + target.y) / 2 + (dx / length) * bow;
    edgeLayer.appendChild(
      node("path", {
        class: "edge",
        d: `M ${source.x} ${source.y} Q ${midX} ${midY} ${target.x} ${target.y}`,
        "stroke-width": 1.1,
        "stroke-opacity": 0.3,
      })
    );
  }
  svg.appendChild(edgeLayer);

  for (const item of hubs) {
    const point = item;
    const group = node("g", { class: seen.has(item.node_id) ? "" : "node-enter" });
    group.appendChild(node("circle", { class: "hub-core", cx: point.x, cy: point.y, r: item.radius }));
    group.appendChild(
      node("circle", {
        class: "hub-ring",
        cx: point.x,
        cy: point.y,
        r: item.radius,
        // Ring thickness reads as "this entity also clusters on other kinds of token".
        "stroke-width": 1.2 + item.shared_kinds * 0.9,
        "stroke-opacity": 0.85,
      })
    );
    const count = node("text", { class: "hub-count", x: point.x, y: point.y + 4 });
    count.textContent = String(item.degree);
    group.appendChild(count);

    const label = node("text", {
      class: "hub-label",
      x: point.x,
      y: point.y + item.label_offset,
    });
    label.textContent = `${item.kind} ${shortToken(item.label)}`;
    group.appendChild(label);
    svg.appendChild(group);
  }

  for (const item of graph.nodes) {
    if (item.kind !== "event") continue;
    const point = item;
    const group = node("g", {
      class: `ev ${item.action} ${seen.has(item.node_id) ? "" : "node-enter"}`,
    });
    if (item.action === "deny") {
      group.appendChild(
        node("circle", {
          class: "ev deny halo", cx: point.x, cy: point.y,
          r: item.radius + 5, "stroke-width": 1.4,
        })
      );
    }
    group.appendChild(node("circle", { cx: point.x, cy: point.y, r: item.radius }));
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${item.label} · ${item.action}`;
    group.appendChild(title);
    if (onSelect && item.decision_id) {
      group.addEventListener("click", () => onSelect(item.decision_id));
    }
    svg.appendChild(group);
  }
}
