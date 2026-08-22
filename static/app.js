// RiskLoom dashboard controller.
//
// Polling, not SSE: events only appear when a preflight is posted, the ledger is small, and there
// is one process. A 3s poll already outpaces the arrival rate, and it has no connection lifecycle
// to get wrong. Polling stops while the tab is hidden so a backgrounded demo does not spin.

import { measure, renderGraph } from "./graph.js";
import {
  actionChip,
  amount,
  clockTime,
  duration,
  el,
  probabilityMarkup,
  shortToken,
  stamp,
} from "./format.js";

const API = "/api/v1/dashboard";
const INTERVALS = { stream: 3000, coordination: 5000, ledger: 10000 };

const state = {
  view: "stream",
  timer: null,
  seenDecisions: new Set(),
  seenNodes: new Set(),
  ledgerFilter: "",
  ledgerSort: { key: "occurred_at", direction: "desc" },
  modelLoaded: false,
};

async function fetchJson(path) {
  const response = await fetch(path, { headers: { accept: "application/json" } });
  if (!response.ok) return { ok: false, status: response.status };
  return { ok: true, data: await response.json() };
}

/* ------------------------------------------------------------------ stream */

function minuteKey(decision) {
  return (decision.occurred_at || decision.created_at || "").slice(0, 16);
}

function decisionRow(decision, isNew, startsMinute) {
  // A rule above the first row of each minute groups a burst visually, so eleven consecutive
  // reviews stop reading as one undifferentiated block.
  const classes = [
    decision.action === "deny" ? "is-deny" : "",
    isNew ? "enter" : "",
    startsMinute ? "minute-start" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const row = el("tr", { class: classes || null, "data-id": decision.decision_id });
  row.innerHTML = `
    <td class="mono muted">${clockTime(decision.occurred_at || decision.created_at)}</td>
    <td class="token">${shortToken(decision.event_id)}</td>
    <td class="token">${shortToken(decision.device_token)}</td>
    <td class="token">${shortToken(decision.network_token)}</td>
    <td class="num">${probabilityMarkup(decision.calibrated_probability, decision.decision_threshold)}</td>
    <td class="num mono">${amount(decision.amount_subunits, decision.currency)}</td>
    <td>${actionChip(decision.action)}</td>`;
  row.addEventListener("click", () => openCase(decision.decision_id));
  return row;
}

async function loadStream() {
  const summary = await fetchJson(`${API}/summary`);
  if (summary.ok) renderStats(summary.data);

  const result = await fetchJson(`${API}/decisions?limit=50`);
  if (!result.ok) return;
  const { decisions, total } = result.data;
  const body = document.getElementById("stream-body");
  body.replaceChildren();
  let previousMinute = null;
  for (const decision of decisions) {
    const minute = minuteKey(decision);
    body.appendChild(
      decisionRow(
        decision,
        !state.seenDecisions.has(decision.decision_id),
        previousMinute !== null && minute !== previousMinute
      )
    );
    previousMinute = minute;
    state.seenDecisions.add(decision.decision_id);
  }
  document.getElementById("stream-empty").hidden = decisions.length > 0;
  document.getElementById("stream-meta").textContent =
    `${decisions.length} of ${total} decisions`;
}

function renderStats(summary) {
  const stats = document.getElementById("stats");
  // Five tiles, each genuinely distinct. "review queue" was dropped: every review action creates
  // exactly one review item, so it always duplicated the "in review" count.
  const cards = [
    ["", "decisions", summary.total_decisions],
    ["allow", "allowed", summary.actions.allow],
    ["review", "in review", summary.actions.review],
    ["deny", "denied", summary.actions.deny],
    ["", "orders created", summary.orders_created],
  ];
  stats.replaceChildren();
  for (const [kind, label, value] of cards) {
    stats.appendChild(
      el("div", { class: `stat ${kind}`.trim() },
        `<div class="label">${label}</div><div class="value">${value}</div>`)
    );
  }
}

/* ------------------------------------------------------------ coordination */

async function loadCoordination() {
  // The panel's rendered size is the one layout input only the browser holds, so it is measured
  // here and forwarded. Everything else about the layout is decided server-side.
  const canvas = document.getElementById("graph");
  const { width, height } = measure(canvas);
  const result = await fetchJson(
    `${API}/coordination?window_seconds=86400&canvas_width=${width}&canvas_height=${height}`
  );
  if (result.ok) {
    const graph = result.data;
    renderGraph(canvas, graph, {
      onSelect: openCase,
      previousIds: state.seenNodes,
    });
    for (const item of graph.nodes) state.seenNodes.add(item.node_id);
    document.getElementById("graph-meta").textContent =
      `${graph.clustered_entity_count} shared entities across ${graph.decision_count} decisions`;
  }
  if (!state.modelLoaded) await loadModelPanel();
}

async function loadModelPanel() {
  const panel = document.getElementById("model-panel");
  const result = await fetchJson(`${API}/model`);
  state.modelLoaded = true;
  if (!result.ok) {
    panel.innerHTML =
      `<div class="empty">Offline evaluation artifact not available in this checkout.</div>`;
    return;
  }
  const { threshold, probability, campaigns, hard_negative_slices: slices } = result.data;
  const pct = (v) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`);
  const rows = [
    ["campaigns", campaigns.campaign_count],
    ["detected", campaigns.detected_campaign_count],
    ["campaign recall", pct(campaigns.campaign_recall)],
    ["missed", campaigns.missed_campaign_count],
    ["—", ""],
    ["recall", pct(threshold.recall)],
    ["precision", pct(threshold.precision)],
    ["false-positive rate", pct(threshold.false_positive_rate)],
    ["average precision", probability.average_precision.toFixed(4)],
    ["ROC-AUC", probability.roc_auc.toFixed(4)],
    ["—", ""],
    ["held-out rows", threshold.row_count.toLocaleString()],
    ["attacks", threshold.attack_count],
  ];
  panel.replaceChildren();
  for (const [key, value] of rows) {
    if (key === "—") {
      panel.appendChild(el("div", { style: "height:1px;background:var(--rl-line);margin:8px 16px" }));
      continue;
    }
    panel.appendChild(
      el("div", { class: "kv tight" }, `<span>${key}</span><span>${value}</span>`)
    );
  }
  panel.appendChild(
    el("div", { class: "ctx-note", style: "padding:8px 16px 14px" },
      "Measured once on the held-out partition with true outcomes. Live serving assumes every " +
      "attempt authorised, which degrades detection — see the known limitation in the README.")
  );
}

/* ------------------------------------------------------------------ ledger */

async function loadLedger() {
  const filter = state.ledgerFilter ? `&action=${state.ledgerFilter}` : "";
  const result = await fetchJson(`${API}/decisions?limit=500${filter}`);
  if (!result.ok) return;
  const { decisions, total } = result.data;

  const { key, direction } = state.ledgerSort;
  const sorted = [...decisions].sort((a, b) => {
    const [x, y] = [a[key], b[key]];
    if (x === y) return 0;
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    const numeric = key === "calibrated_probability" || key === "amount_subunits";
    const compared = numeric ? Number(x) - Number(y) : String(x).localeCompare(String(y));
    return direction === "asc" ? compared : -compared;
  });

  const body = document.getElementById("ledger-body");
  body.replaceChildren();
  for (const d of sorted) {
    const row = el("tr", { class: d.action === "deny" ? "is-deny" : null });
    row.innerHTML = `
      <td class="mono muted">${stamp(d.occurred_at || d.created_at)}</td>
      <td class="token">${shortToken(d.event_id)}</td>
      <td class="token">${shortToken(d.merchant_id)}</td>
      <td class="token">${shortToken(d.device_token)}</td>
      <td class="token">${shortToken(d.payment_instrument_token)}</td>
      <td class="num">${probabilityMarkup(d.calibrated_probability, d.decision_threshold)}</td>
      <td class="num mono">${amount(d.amount_subunits, d.currency)}</td>
      <td class="mono muted">${d.risk_decision || "—"}</td>
      <td>${actionChip(d.action)}</td>
      <td class="mono muted" style="font-size:11px">${d.fail_safe_reason || "—"}</td>
      <td class="token">${d.razorpay_order_id || "—"}</td>`;
    row.addEventListener("click", () => openCase(d.decision_id));
    body.appendChild(row);
  }
  document.getElementById("ledger-empty").hidden = sorted.length > 0;
  document.getElementById("ledger-meta").textContent = `${sorted.length} of ${total}`;
}

/* ------------------------------------------------------------- case detail */

async function openCase(decisionId) {
  const result = await fetchJson(`${API}/decisions/${decisionId}`);
  if (!result.ok) return;
  const { decision: d, context, review_pending: pending } = result.data;

  const probability = Number(d.calibrated_probability);
  const threshold = Number(d.decision_threshold);
  const delta = probability - threshold;
  const side = delta >= 0 ? "at or above" : "below";

  const contextRows = context
    .map((c) => {
      if (!c.token) {
        return `<div class="ctx"><span class="k">${c.kind}</span><span class="v muted">absent</span></div>`;
      }
      const denied = c.denied_count > 0 ? `<em>${c.denied_count} denied</em>` : "0 denied";
      return `<div class="ctx">
        <span class="k">${c.kind} <span class="token">${shortToken(c.token)}</span></span>
        <span class="v">${c.decision_count} decisions · ${duration(c.span_seconds)} · ${denied}</span>
      </div>`;
    })
    .join("");

  document.getElementById("case-body").innerHTML = `
    <div class="verdict">
      <div>
        <div class="event">${d.event_id}</div>
        <div class="muted" style="font-size:12px;margin-top:4px">
          ${stamp(d.occurred_at || d.created_at)} · ${d.channel} · ${amount(d.amount_subunits, d.currency)}
        </div>
      </div>
      <div class="action ${d.action}">${d.action}</div>
    </div>

    <div class="compare">
      <div class="row"><span class="tag">probability</span>
        ${probabilityMarkup(d.calibrated_probability, d.decision_threshold)}</div>
      <div class="row"><span class="tag">threshold</span>
        <span class="prob"><i>${d.decision_threshold}</i></span></div>
      <div class="delta">${side} the locked threshold by ${Math.abs(delta).toExponential(3)}
        ${d.fail_safe_reason ? `· action downgraded to review: <b>${d.fail_safe_reason}</b>` : ""}</div>
    </div>

    <div class="detail-grid">
      <div>
        <h3>Stored attributes</h3>
        <div class="ctx"><span class="k">risk decision</span><span class="v">${d.risk_decision || "—"}</span></div>
        <div class="ctx"><span class="k">device</span><span class="v token">${shortToken(d.device_token)}</span></div>
        <div class="ctx"><span class="k">network</span><span class="v token">${shortToken(d.network_token)}</span></div>
        <div class="ctx"><span class="k">instrument</span><span class="v token">${shortToken(d.payment_instrument_token)}</span></div>
        <div class="ctx"><span class="k">session</span><span class="v token">${shortToken(d.session_token)}</span></div>
        <div class="ctx"><span class="k">merchant</span><span class="v token">${shortToken(d.merchant_id)}</span></div>
        <div class="ctx"><span class="k">order</span><span class="v token">${d.razorpay_order_id || "—"}</span></div>
        <div class="ctx"><span class="k">model</span><span class="v token">${shortToken(d.model_id)}</span></div>
        <div class="ctx"><span class="k">review queued</span><span class="v">${pending ? "yes" : "no"}</span></div>
      </div>
      <div>
        <h3>Ledger co-occurrence</h3>
        ${contextRows}
        <div class="ctx-note">
          Counted from decisions already written to the ledger. These are not the model's
          features — the feature vector is not persisted, and recomputing it would produce values
          this decision never saw.
        </div>
      </div>
    </div>`;

  show("case");
}

/* -------------------------------------------------------------- view logic */

function show(view) {
  state.view = view;
  for (const name of ["stream", "coordination", "ledger", "case"]) {
    document.getElementById(`view-${name}`).hidden = name !== view;
  }
  for (const button of document.querySelectorAll("nav.views button")) {
    button.setAttribute("aria-selected", String(button.dataset.view === view));
  }
  const label = document.getElementById("live-label");
  const live = document.getElementById("live-state");
  if (INTERVALS[view]) {
    label.textContent = `live · ${INTERVALS[view] / 1000}s`;
    live.classList.remove("paused");
  } else {
    label.textContent = "detail";
    live.classList.add("paused");
  }
  restartPolling();
}

const LOADERS = { stream: loadStream, coordination: loadCoordination, ledger: loadLedger };

function restartPolling() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  const loader = LOADERS[state.view];
  if (!loader) return;
  loader();
  if (document.hidden) return;
  state.timer = setInterval(loader, INTERVALS[state.view]);
}

document.addEventListener("visibilitychange", restartPolling);
document.getElementById("case-back").addEventListener("click", () => show("stream"));
for (const button of document.querySelectorAll("nav.views button")) {
  button.addEventListener("click", () => show(button.dataset.view));
}
document.getElementById("ledger-filter").addEventListener("change", (event) => {
  state.ledgerFilter = event.target.value;
  loadLedger();
});
for (const header of document.querySelectorAll("th.sortable")) {
  header.addEventListener("click", () => {
    const key = header.dataset.sort;
    state.ledgerSort =
      state.ledgerSort.key === key
        ? { key, direction: state.ledgerSort.direction === "asc" ? "desc" : "asc" }
        : { key, direction: "desc" };
    loadLedger();
  });
}

show("stream");
