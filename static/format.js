// Presentation helpers. No risk logic lives here: every number rendered was computed elsewhere
// and is shown exactly as the API returned it.

/** Shorten a pseudonymous token for display without altering it. */
export function shortToken(token) {
  if (!token) return "—";
  const [prefix, body] = [token.slice(0, 4), token.slice(4)];
  return `${prefix}…${body.slice(-6)}`;
}

export function clockTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

export function stamp(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${clockTime(iso)}`;
}

export function amount(subunits, currency) {
  const major = (Number(subunits) / 100).toFixed(2);
  return `${major} ${currency}`;
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m < 60 ? `${m}m ${s}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

// A value is only marked when it is close enough to the threshold that the exact digits matter.
// Everything else — the vast majority of rows — renders as a plain uniform number.
const NEAR_THRESHOLD_RATIO = 0.25;

/**
 * Render a probability at full precision, marking at most one digit.
 *
 * Full precision is kept because the locked threshold carries nineteen significant decimals and a
 * real tie-cluster of scored rows sits one unit in the last place below it, so rounding for
 * display would erase the distinction the system is built around.
 *
 * The marking is deliberately minimal: a single accent-coloured character at the first position
 * where this value's decimals diverge from the threshold's, and only when the value is within
 * NEAR_THRESHOLD_RATIO of the threshold. A boundary case therefore shows one tinted digit, and an
 * ordinary row shows none at all.
 */
export function probabilityMarkup(probability, threshold) {
  if (probability === null || probability === undefined) return '<span class="muted">—</span>';

  const value = Number(probability);
  const limit = Number(threshold);
  const near =
    Number.isFinite(value) &&
    Number.isFinite(limit) &&
    limit > 0 &&
    Math.abs(value - limit) / limit <= NEAR_THRESHOLD_RATIO;

  if (!near) return `<span class="prob">${probability}</span>`;

  let index = 0;
  while (
    index < probability.length &&
    index < threshold.length &&
    probability[index] === threshold[index]
  ) {
    index += 1;
  }
  // Identical to the threshold: there is no diverging digit, but this is the boundary case
  // itself, so the final digit is marked rather than showing nothing at all.
  if (index >= probability.length) {
    const head = probability.slice(0, -1);
    const last = probability.slice(-1);
    return `<span class="prob">${head}<b class="divergent">${last}</b></span>`;
  }

  const head = probability.slice(0, index);
  const marked = probability[index];
  const tail = probability.slice(index + 1);
  return `<span class="prob">${head}<b class="divergent">${marked}</b>${tail}</span>`;
}

export function actionChip(action) {
  return `<span class="chip ${action}">${action}</span>`;
}

export function el(tag, attrs = {}, html = "") {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (html) node.innerHTML = html;
  return node;
}
