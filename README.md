# RiskLoom

**A defense-only, shadow-mode AI risk manager for coordinated card-testing fraud.**
Built for the Razorpay AI Buildathon, Track 02.

Card testing is defined by *reuse*: one attacker driving many small authorization attempts across
shared devices, networks and instruments to find live cards. RiskLoom detects that shape. It scores
a live checkout attempt against a locked, calibrated model using 75 causal features computed from
event history, writes an append-only audit ledger, and surfaces the result on an operations
dashboard with an LLM-written explanation of *why*.

It is shadow-mode by construction. RiskLoom never captures, refunds, or blocks a payment. A DENY
means no Razorpay order is created; nothing else in the payment flow is touched. Every external
integration is test-mode only, and live-mode API keys are rejected at startup.

---

## Held-out results

Measured once, on a held-out partition the model had never seen, through the same portable
inference path the live service uses:

| Metric | Value |
| --- | ---: |
| Recall | **0.9765** |
| Precision | **0.7685** |
| Average precision | **0.9640** |
| ROC-AUC | 0.9886 |
| False-positive rate | **0.60%** |
| Cost (FN×25 + FP×1) | 300 units |
| Rows / attacks | 17,000 / 340 |

**Read that number honestly.** It describes detection of attacks shaped like the training data. A
later gate deliberately attacked the detector's own mechanisms and drove recall to between 0.00 and
0.12 — see [Known limitations](#known-limitations-and-honest-disclosures), which leads with that
finding rather than burying it.

The data is synthetic. RiskLoom makes no production fraud-accuracy claim.

---

## Architecture

```
                          ┌──────────────────────────────────────────┐
   Razorpay test-mode ───▶│ webhook ingest  HMAC · idempotent · redacted
                          └──────────────────────────────────────────┘
                                            │ append-only observations
                                            ▼
  ┌────────────────┐   events    ┌────────────────────┐   75 features
  │  simulator     │────────────▶│  causal feature    │───────────────┐
  │  deterministic │             │  engine  (Day 3)   │               │
  └────────────────┘             └────────────────────┘               ▼
                                       compute-before-update   ┌──────────────┐
                                       sliding 60/300/3600s    │ locked model │
                                                               │  + threshold │
                                                               └──────────────┘
                                                                      │
   POST /checkout/preflight ──▶ claim → score → act → finalise ───────┘
        │                        (idempotent, fail-safe to REVIEW)
        │                                   │
        │                                   ├──▶ Razorpay order (ALLOW only, capped)
        │                                   └──▶ risk_decisions ledger (append-only)
        ▼                                              │
   ALLOW / REVIEW / DENY                               ▼
                                    ┌───────────────────────────────────┐
                                    │ dashboard (read-only)             │
                                    │  stream · coordination · ledger   │
                                    │  case detail + LLM explanation    │
                                    │  PSI drift (informational)        │
                                    └───────────────────────────────────┘

   offline only, unreachable from any decision:
     policy engine (built, neither candidate approved) · adversarial stress · blind-spot analysis
```

Isolation is enforced, not intended. Static AST checks plus fresh-interpreter probes assert that
the decision path cannot reach the policy engine, the drift module, the explanation generator or the
offline analysis packages — and that none of them can reach it.

---

## Run it

**You need two things: this repository, and the generated-artifact bundle.**

A `git clone` alone **cannot** start the service. The locked model, its manifest and the feature
manifest are deliberately excluded from version control — committing them would misrepresent
generated artifacts as source, and regenerating them elsewhere would produce a *different* model and
break every hash in this document. They are inputs, supplied alongside the clone.

```powershell
# 1. Confirm the artifact bundle is in place. Names exactly what is missing if it is not.
uv run python scripts/preflight_check.py

# 2. Configure. The placeholders are valid for startup; real Razorpay test keys are only
#    needed to create actual orders, and the Gemini key is optional.
Copy-Item .env.example .env

# 3. Bring up the full stack: postgres → migrations → app.
docker compose up --build -d

# 4. Open the dashboard.
Start-Process http://127.0.0.1:8000/dashboard
```

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker Desktop with Linux containers. Python 3.11
is fetched by uv; no local PostgreSQL client is required.

<details>
<summary>Port already in use, or running without Docker</summary>

Host ports `5432` and `8000` must be free. If either is taken, set both values in `.env` — Compose
and the application must agree:

```dotenv
RISKLOOM_DATABASE_URL=postgresql+asyncpg://riskloom:riskloom_local_only@127.0.0.1:5433/riskloom
RISKLOOM_POSTGRES_PORT=5433
RISKLOOM_APP_PORT=8001
```

Keep `.env.example` at the documented defaults and use `.env` for machine-specific overrides.

To run the app on the host instead of in a container, start only the database and run the
migrations and server yourself:

```powershell
docker compose up -d --wait postgres
uv run alembic upgrade head
uv run uvicorn riskloom.main:app --port 8000
```

On Windows, clone into a short path such as `C:\riskloom`. A deep path can exceed the 260-character
limit and break loading of compiled dependencies.
</details>

### Score a checkout

```powershell
$body = @{
  event_id = "evt_00000000000000000000000000000f01"
  merchant_id = "mrc_00000000000000000000000000000001"
  checkout_id = "chk_00000000000000000000000000000f01"
  customer_token = $null
  device_token = "dev_00000000000000000000000000000f01"
  network_token = "net_00000000000000000000000000000f01"
  session_token = "ses_00000000000000000000000000000f01"
  payment_instrument_token = "pmt_00000000000000000000000000000f01"
  amount_subunits = 25000; currency = "INR"; channel = "web"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/checkout/preflight `
  -ContentType application/json -Body $body
```

Every token is a pseudonymous `prefix_<32 hex>` value; the request schema rejects anything else, and
rejects any field that looks like PII. Repeat the call with the same `event_id` to see idempotency:
the second response carries `"duplicate": true` and creates no second effect.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health/live` | process only |
| GET | `/health/ready` | PostgreSQL connectivity |
| POST | `/api/v1/webhooks/razorpay` | signed Razorpay webhooks |
| POST | `/api/v1/checkout/preflight` | score one live checkout attempt |
| GET | `/api/v1/dashboard/summary` | ledger counts and provenance |
| GET | `/api/v1/dashboard/decisions` | paged decision rows |
| GET | `/api/v1/dashboard/decisions/{id}` | one decision + ledger context |
| GET | `/api/v1/dashboard/coordination` | shared-token graph |
| GET | `/api/v1/dashboard/model` | offline held-out aggregates |
| GET | `/api/v1/dashboard/drift` | PSI against the locked reference |
| GET | `/api/v1/dashboard/decisions/{id}/explanation` | read a generated explanation |
| POST | `/api/v1/dashboard/decisions/{id}/explanation` | generate one |

There is deliberately no public order-creation endpoint; the Orders client is an internal adapter.
The dashboard is read-only apart from that single explanation POST, which writes one enrichment row
and can alter no decision. Every other verb on every dashboard path answers 405 by routing.

---

## Known limitations and honest disclosures

Every limitation below was found by measuring this system rather than by reasoning about it, and
each was published in the gate that found it. They are collected here so the picture is in one
place. A system that knows precisely what it cannot do is the point, not an apology.

### 1. The model is evadable by boundary-spaced traffic — recall 0.9765 → 0.00

Attack traffic shaped to defeat the detector's own mechanisms drove recall from 0.9765 to between
0.00 and 0.12, with cost rising roughly tenfold. Spacing attempts at *exactly* one feature window
(3600s) achieved **complete evasion**: 0 of 120 events and 0 of 20 campaigns detected, with mean
attack probability **0.0018 against 0.0092 for legitimate traffic** — the model rated the evasive
attacks as *less* risky than ordinary customers.

The cause is specific and mechanistic, not a general failure. Rolling windows use a left-exclusive
`(t - w, t]` cutoff, so an event exactly one window back is already expired; spacing at the window
length places every prior event on the excluded boundary and every attempt counter reads zero. The
same model, on the same file, detects baseline-shaped attacks at **83–87%**.

That figure is the exploit's theoretical ceiling — deterministic spacing, no jitter, full knowledge
of the feature schema — and is **not** what an evasive attacker generally achieves. The fix is
understood and deliberately not implemented: staggered windows, a longer-horizon window that
boundary spacing cannot evade simultaneously, or treating suspiciously regular timing as its own
signal. Any of those needs a schema increment and a re-locked model, which is its own gate.

→ [Full analysis](docs/BUILD_LOG.md#gate-h0-adversarial-stress-test)

### 2. Live-serving accuracy is not measured to held-out standard

At preflight an attempt's outcome does not exist yet, so live serving advances feature state with
every attempt recorded as authorized. The 57 outcome-independent features are exact; the 18
failure-derived ones read low. Quantified, not assumed — replaying 9,000 events under both
assumptions gives recall 0.856 → 0.600, precision 0.577 → 0.184, FPR 1.28% → 5.43%, and cost 763 →
2,279 (**+199%**). Held-out numbers must not be quoted as live-serving accuracy.

→ [Detail](docs/BUILD_LOG.md#known-limitation-live-serving-accuracy-is-not-measured-to-held-out-standard)

### 3. The locked threshold sits one ULP above a tie cluster

The model produces only 15 distinct calibrated probabilities across 7,952 selection rows, so
probabilities arrive in large ties. The locked threshold lands one unit in the last place above a
cluster of eight rows, so portable inference allows rows the training report recorded as denied. No
published held-out number is affected — Gate B2 computed every figure through portable inference.
Monitored by a non-fatal `validate-model` diagnostic rather than silently re-locked.

→ [Detail](docs/BUILD_LOG.md#decision-boundary-diagnostic)

### 4. Calibration is overconfident in the mid probability range

The 0.4–0.8 reliability bins predict materially higher risk than they observe. Expected calibration
error is low overall because almost all mass sits in the lowest bin, which hides the mid-band error.

### 5. Drift detection is a coarse instrument

PSI is computed against the locked held-out reference, which is degenerate for the purpose: 97.78%
of rows fall in one bin, five bins are empty, and the decision threshold itself sits *inside* the
most populated bin, so the binning has almost no resolution where decisions happen. The zero-bin
epsilon is load-bearing — the same data reads 0.0559 ("no shift") at 1e-3 and 0.2122 ("moderate") at
1e-6. The surface therefore reports no band at all below 200 rows and always shows per-bin
contributions.

### 6. A frozen database is not bounded server-side

A database that *refuses* connections fails preflight in about 3s with 503. One that is *frozen*
keeps its socket open and answers nothing, and neither the connect timeout nor an `asyncio.timeout`
fires — SQLAlchemy's greenlet bridge does not deliver cancellation into asyncpg's blocked read, so
the request unwinds only when the connection dies. It still answers 503 and never an unbacked
ALLOW, so the decision stays fail-closed; the latency is not bounded.

### 7. An orphaned Razorpay order is possible

If storage dies after an ALLOW created an order, the order exists with no ledger record. The caller
gets 503, the row stays `pending`, and the order id is logged under `preflight_ledger_write_failed`.
There is no auto-recovery, by design.

### 8. Neither cost-aware policy band was approved

The banded policy beat the incumbent on cost but exceeded the configured false-positive-rate
ceiling, so approval was refused and the incumbent single threshold remains in force. The comparison
is published whether the policy wins or loses.

### 9. Operational limits

- **No authentication.** Every endpoint is unauthenticated; acceptable only for a local build.
- **Live feature state is in-memory.** It does not survive a restart; history features read zero
  until traffic rebuilds them.
- **Review items cannot be worked.** They are recorded and counted, never resolved or overridden.
- **Explanations are enrichment, never evidence.** Generated after the fact from stored aggregates,
  with no path to any decision. Free-tier terms may permit training on submitted content, which is
  precisely why the input contract carries only aggregates and enums.
- **The coordination graph is not campaign detection.** It is a projection of shared stored tokens;
  no model decides that those decisions form a campaign.

---

## Verification

Everything below is expected to pass on a clean checkout with the artifact bundle in place.

```powershell
uv sync
uv run ruff format --check . ; uv run ruff check . ; uv run mypy src
uv run alembic upgrade head ; uv run alembic downgrade base
uv run alembic upgrade head ; uv run alembic check
uv run pytest --cov=riskloom --cov-report=term-missing --cov-fail-under=90
```

897 tests, 90% branch coverage. Integration tests start a disposable PostgreSQL 16 container and
never call Razorpay or Gemini; every external call in the suite is faked.

The four locked artifacts are pinned and verified at every gate:

| Artifact | SHA-256 |
| --- | --- |
| `model.json` | `3db8dafef643261c0df559cab632cfaf6fc45be54f38c1f4a621ef5af84039d4` |
| `training_report.json` | `4ff96556f1df49d7c44c29703c328753a6ea0ef197410976100e84751943ea6d` |
| `manifest.json` | `00aa16380eee1dcfa26fe9c89ed0eb8f866e75e98bd7a7ba89f9cc228c792f2e` |
| `evaluation.json` | `11251cef0dade5d14d2d1a85fe3822126e01c2a354494dd09b720c679244c40d` |

---

## Deeper reading

| Document | Contents |
| --- | --- |
| [docs/BUILD_LOG.md](docs/BUILD_LOG.md) | The day-by-day engineering record: every design decision, what was measured, what was rejected and why |
| [AGENTS.md](AGENTS.md) | Durable engineering instructions and invariants for anyone changing this codebase |
| [.env.example](.env.example) | The complete configuration surface |

Settings use the `RISKLOOM_` prefix and secrets are `SecretStr` values that are never interpolated
into logs or exceptions. Webhook bodies are verified against their exact raw bytes, hashed, and
discarded; only an allowlisted projection is ever stored. Fixtures use reserved synthetic values
only.
