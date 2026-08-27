<div align="center">

# RiskLoom

### Detect coordinated card testing before a checkout becomes a loss

[![Razorpay Buildathon](https://img.shields.io/badge/Razorpay_AI_Buildathon-Track_02-0B72E7?style=flat-square)](https://razorpay.com/buildathon)
![Held-out recall](https://img.shields.io/badge/held--out_recall-97.6%25-16803A?style=flat-square)
![False-positive rate](https://img.shields.io/badge/false--positive_rate-0.60%25-16803A?style=flat-square)
![Defense only](https://img.shields.io/badge/scope-defense_only-5C2D91?style=flat-square)

**Live checkout scoring · campaign coordination · explainable decisions · append-only audit trail**

[Results](#held-out-evaluation) · [Architecture](#system-architecture) · [Run locally](#run-locally) · [Safety](#safety-by-design) · [Limitations](#what-testing-revealed)

</div>

## 30-second judge card

**Merchant-loss problem:** low-value card-testing attempts look ordinary one by one, so merchants
need coordinated abuse detected before repeated checkout attempts become losses.

**RiskLoom:** a defense-only, shadow-mode risk manager that turns 75 causal temporal features into
an auditable checkout risk decision; Gemini remains post-decision, and the system never captures or
modifies payments.

| Judge question | Verified answer |
| --- | --- |
| Track | **Track 02 — AI Risk Manager** |
| Held-out proof | **17,000** synthetic chronological test events; **340** attacks; campaign recall **3/3 (100%)**; event recall **332/340 (97.65%)** |
| Decision quality | Precision **76.85%**; average precision **96.40%**; ROC-AUC **98.86%** |
| False-positive burden | **100/16,660 legitimate events (0.60%)**, with **16,560** true negatives; costs are abstract policy units, never rupee savings |
| AI judgment | The locked calibrated model makes the `ALLOW`/`DENY` risk decision; `REVIEW` is an operational fail-safe. Gemini only explains an already-final `DENY` and cannot alter its score, threshold, decision, action, or payment behavior. |
| Safety boundary | Defense-only; synthetic data; Razorpay test mode; no capture, refund, settlement, payment modification, or public order-creation endpoint |

**Verify:** run [`uv run python scripts/preflight_check.py`](scripts/preflight_check.py), then follow
the [evaluator evidence map](docs/JUDGING.md). See the [architecture](docs/ARCHITECTURE.md),
[held-out evidence](docs/BUILD_LOG.md#day-4-offline-model-locking), [safety boundary](#safety-by-design),
[reproduction path](#run-locally), and strict [submission manifest](submission_manifest.json).

**Baseline relationship:** `v1.0.3-submission` at
`ff36ba091f5bcf44b45e40044139996663f03bca` is the immutable engineering and runtime baseline
immediately preceding this evaluator-readiness package. This package changes documentation,
machine-readable metadata, and validation tests only and is suitable for a new source submission
release; it does not change the model, runtime bundle, artifact hashes, features, APIs, schema,
dependencies, or inference behavior.

<p align="center">
  <img src="docs/assets/riskloom-coordination.png" alt="RiskLoom coordination dashboard connecting checkout decisions through a shared device and network" width="100%">
</p>

<p align="center"><sub>A shared-signal view of 32 live decisions, with locked held-out model metrics shown separately from live traffic.</sub></p>

## Why RiskLoom

Card-testing attempts often look harmless in isolation. The amounts are small, the instruments keep changing, and each checkout can resemble an ordinary failure. The pattern becomes visible only when activity is connected across devices, sessions, networks, instruments, merchants, and time.

RiskLoom finds that coordination while a checkout is still in progress. It computes 75 causal temporal features, scores the attempt with a locked calibrated model, returns `ALLOW`, `REVIEW`, or `DENY`, and records the evidence behind the outcome. Operators can watch the pattern form in real time without giving an LLM or dashboard control over the decision.

RiskLoom is defense-only and shadow-mode by construction. It uses synthetic data and Razorpay test mode. It never captures, refunds, settles, or modifies a payment.

## System Architecture

RiskLoom keeps deterministic model judgment, operational safety, external Razorpay test-mode
operations, generative explanations, and append-only audit evidence in separate trust boundaries.
The locked model makes only `ALLOW` or `DENY` risk decisions; `REVIEW` is an operational safety
outcome when execution cannot safely complete. Gemini operates after a finalized `DENY`, outside
the decision boundary, and cannot change the score, threshold, decision, action, or payment
behavior. See the complete [system architecture, trust-boundary table, and failure
model](docs/ARCHITECTURE.md).

## What it does

- Scores a checkout before Razorpay order creation.
- Detects repeated infrastructure and coordinated activity over time.
- Keeps feature computation causal through a compute-before-update state engine.
- Loads the selected model from strict JSON rather than pickle or joblib.
- Writes every final decision to an append-only audit ledger.
- Shows decisions, shared-token coordination, model evaluation, and drift diagnostics on one dashboard.
- Generates explanations only after a decision is final.

An `ALLOW` can create a capped Razorpay test-mode order. `REVIEW` and `DENY` create no order. There is deliberately no public order-creation endpoint.

## Held-out evaluation

The final model, calibrator, feature order, threshold, and cost policy were locked before the chronological test period was opened. Nothing was refit after evaluation.

| Metric | Result |
| --- | ---: |
| Test rows | 17,000 |
| Attack events | 340 |
| Campaign recall | **100% (3/3)** |
| Event recall | **97.65%** |
| Precision | **76.85%** |
| Average precision | **96.40%** |
| ROC-AUC | **98.86%** |
| False-positive rate | **0.60%** |
| Configured policy cost | **300 units** |

### Confusion matrix

| Actual class | Predicted attack | Predicted legitimate |
| --- | ---: | ---: |
| Attack | **332** | 8 |
| Legitimate | 100 | **16,560** |

The policy cost is `false positives × 1 + false negatives × 25`. These are transparent decision weights, not estimates of merchant losses in rupees.

All results come from deterministic synthetic data. They demonstrate the behavior of this implementation under its documented data model, not production fraud accuracy.

## Product walkthrough

### A coordinated burst changes the decision

The dashboard can post a local synthetic burst through the same preflight endpoint used by normal requests. Reuse accumulates across the shared device and network, and the sequence moves from `ALLOW` to `REVIEW` and then `DENY`.

![RiskLoom live decision stream showing an attack burst progressing from allow to review and deny](docs/assets/riskloom-live-stream.png)

### A decision remains inspectable

The detail view separates the locked model result from the final action, lists only stored pseudonymous attributes, and shows prior ledger co-occurrence. The explanation is generated from stored aggregates after the decision is final and cannot change it.

![RiskLoom denied decision with probability, threshold, shared-signal evidence, and generated explanation](docs/assets/riskloom-decision-deny.png)

<details>
<summary>View ALLOW and REVIEW decision examples</summary>

#### Allowed checkout

The model score remains below the locked threshold and a capped test-mode order is created.

![RiskLoom allowed checkout with test-mode Razorpay order](docs/assets/riskloom-decision-allow.png)

#### Review fallback

The model result is preserved separately from the operational action. In this example, order-budget exhaustion safely downgrades the action to `REVIEW`.

![RiskLoom review decision caused by order budget exhaustion](docs/assets/riskloom-decision-review.png)

</details>

### The ledger preserves the final record

The read-only ledger brings together the model risk, final action, fail-safe reason, test order reference, and pseudonymous entity tokens. Repeated requests remain idempotent rather than creating a second effect.

![RiskLoom append-only audit ledger containing allow, review, and deny decisions](docs/assets/riskloom-audit-ledger.png)

## How it works

```text
Razorpay test-mode checkout
          │
          ▼
  Checkout preflight API
          │
          ├── validate request and merchant scope
          ├── resolve prior temporal state
          └── compute 75 causal features
                         │
                         ▼
                 Locked JSON model
                 + Platt calibrator
                 + locked threshold
                         │
                         ▼
                ALLOW / REVIEW / DENY
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Append-only ledger       Explanation job
             │                 current decision excluded
             ├───────────────┐       │
             ▼               ▼       ▼
       Operator dashboard  Coordination graph  LLM explanation
```

Four boundaries keep the result reproducible and auditable:

1. **Causal state:** an event is scored from observations that existed before it; the current event is added only afterward.
2. **Chronological isolation:** training, calibration fitting, policy selection, and final test are separate periods with separate interfaces.
3. **Portable inference:** the model is validated JSON data with no arbitrary code-loading surface.
4. **Explanation isolation:** Gemini receives allowlisted aggregates after the outcome and has no path back to scoring or payment behavior.

## Run locally

### Requirements

- Docker Desktop with Compose
- [`uv`](https://docs.astral.sh/uv/)
- Razorpay test credentials only for creating actual test-mode orders
- Optional Gemini API access for generated explanations

The locked model and its manifests are generated artifacts, deliberately excluded from Git:
committing them would misrepresent generated output as source, and regenerating them elsewhere
would produce a *different* model and invalidate every hash published here. They are
distributed as a hash-pinned release bundle instead, which
`runtime_bundle.py install` downloads and verifies.

The bundle is published at
[**`v1.0.3-runtime`**](https://github.com/XxSURYANSHxX/riskloom/releases/tag/v1.0.3-runtime), and
this is the normal path — no manual download, no `--archive`.

```powershell
git clone https://github.com/XxSURYANSHxX/riskloom.git
cd riskloom

uv sync --frozen
uv run python scripts/runtime_bundle.py install
uv run python scripts/preflight_check.py

Copy-Item .env.example .env
docker compose up --build -d
```

The installer downloads one fixed release asset, checks every member against a SHA-256 pinned
in source, writes only the five approved paths, and then runs the application's own startup
binding, so a bundle that installs but cannot serve is reported as a failure. It accepts no URL
argument and reads no environment override.

Already hold a verified archive — an air-gapped machine, a mirrored copy? Install it offline
instead. This is a secondary path; the command above is the normal one:

```powershell
uv run python scripts/runtime_bundle.py install `
  --archive C:\path\to\riskloom-runtime-artifacts.zip
```

Build the archive yourself from an existing installation:

```powershell
uv run python scripts/runtime_bundle.py build `
  --output dist/riskloom-runtime-artifacts.zip
```

Verify an existing installation at any time:

```powershell
uv run python scripts/runtime_bundle.py verify --require-evaluation
```

**Verified from a clean clone on 2026-08-26.** A public HTTPS clone with no `artifacts/`
directory: `uv sync --frozen` succeeded, preflight correctly failed before installation and named
all four startup artifacts, then plain `install` — no `--archive` — downloaded the published release
and installed five artifacts with zero hash mismatches. A second install reported all five
unchanged, and preflight then passed. `/health/live`, `/health/ready`, `/dashboard`,
`/api/v1/dashboard/model` and `/api/v1/dashboard/drift` all returned HTTP 200 from an isolated
Compose stack. Existing Docker resources were not modified. Full evidence is in the
[build log](docs/BUILD_LOG.md).

<details>
<summary>What the bundle contains, and what it never contains</summary>

The published ZIP is **58,070 bytes**, SHA-256
`5f789aecdd74ab31a92cfdb9da5d8d1312e89ac488b79a763374b5e425046cfe`. The ZIP contains six canonical
JSON members: five approved runtime artifacts plus one bundle manifest. The manifest validates the
archive and is not installed, so archive members = 6 and installed artifacts = 5. The five artifacts
total 55,743 bytes:

| Path | Purpose |
| --- | --- |
| `artifacts/models/development/model.json` | the locked model and its threshold |
| `artifacts/models/development/training_report.json` | required by the strict model-directory check |
| `artifacts/models/development/manifest.json` | model identity and source contract |
| `artifacts/features/development-v1.1.0-config-bound/manifest.json` | feature-configuration binding |
| `artifacts/evaluations/development/evaluation.json` | held-out aggregates, drift reference, model panel |

The first four are required to start. The fifth is required for the complete product: without it the
service still starts and scores normally, but the dashboard's model panel answers 404 and the drift
endpoint has no reference.

It contains **no** simulation events, labels, feature rows, feature reports, per-event predictions,
campaign or event identifiers, pseudonymous entity tokens, raw payment data, credentials, `.env`,
API keys, webhook secrets, PII, database contents, caches, or logs — and no pickle, joblib, or other
executable serialised Python. Every member is aggregate JSON this repository already publishes
hashes for.

Two independent layers of provenance, and both must hold. The complete ZIP is pinned in source and
checked **before any parsing at all**, so a reshaped or tampered file meets a hash comparison rather
than format-parsing code; the five member hashes are pinned separately and remain authoritative on
their own. Installation is idempotent — a second run reports all five unchanged and rewrites
nothing — and fail-closed: a bundle that installs but cannot serve is reported as a failure.

Runtime and submission snapshots use separate immutable tags: `v1.0.3-runtime` for the published
runtime asset and `v1.0.3-submission` for the final verified source snapshot. Published tags are
never moved; later versions use new tags. A published bundle's manifest records the tag it was
released under, so moving that tag would make an already-distributed artifact describe a commit it
did not come from.

</details>

Open the dashboard:

```text
http://127.0.0.1:8000/dashboard
```

Check readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

The placeholders in `.env.example` are sufficient for startup. Real Razorpay test credentials are needed only when an `ALLOW` should create a test order. Never commit `.env`.

<details>
<summary>API surface</summary>

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | PostgreSQL readiness |
| `POST` | `/api/v1/webhooks/razorpay` | Verified Razorpay webhook intake |
| `POST` | `/api/v1/checkout/preflight` | Score one checkout attempt |
| `GET` | `/api/v1/dashboard/summary` | Decision and outcome summary |
| `GET` | `/api/v1/dashboard/decisions` | Paged audit ledger |
| `GET` | `/api/v1/dashboard/decisions/{id}` | One decision with ledger context |
| `GET` | `/api/v1/dashboard/coordination` | Shared-token graph |
| `GET` | `/api/v1/dashboard/model` | Locked model metadata and held-out aggregates |
| `GET` | `/api/v1/dashboard/drift` | Informational PSI diagnostics |
| `GET` | `/api/v1/dashboard/decisions/{id}/explanation` | Read an explanation |
| `POST` | `/api/v1/dashboard/decisions/{id}/explanation` | Generate one explanation enrichment row |

</details>

## Safety by design

- Live-mode Razorpay keys are rejected at startup.
- Webhooks are verified against their exact raw bytes before parsing.
- Stored identifiers are strict synthetic or pseudonymous tokens.
- Raw card details, credentials, full payloads, and PII are never sent to Gemini.
- LLM output is explanation only and cannot become model evidence or decision authority.
- Adversarial testing is local, synthetic, and unable to contact an external target.
- Database failures never produce an unbacked `ALLOW`.
- The dashboard is read-only apart from explanation enrichment, which cannot alter a decision.

## What testing revealed

The complete measurements and failure narratives are preserved in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md). The most important open boundaries are summarized here.

| Test | Finding | Current response |
| --- | --- | --- |
| Boundary-spaced evasion | Recall fell to **0.00-0.12**; exact 3,600-second spacing detected **0/120** attack events | Kept the locked model unchanged, published the failure, and scoped longer-horizon and spacing-aware features for a future schema |
| Live-state replay | Recall moved from **0.856 to 0.600** and policy cost from **763 to 2,279** under live-serving assumptions | Offline held-out metrics are never presented as measured live accuracy |
| Threshold ties | The locked threshold sits one floating-point step above an eight-row probability cluster | Portable inference parity and a non-fatal boundary diagnostic are tested |
| Midrange calibration | Middle probability bands are overconfident despite low overall ECE | Reliability tables remain visible and no test-tuned recalibration was performed |
| Candidate policy | A lower-cost banded policy exceeded the false-positive ceiling | Activation was refused and the incumbent threshold remained locked |
| Frozen database | Failure remains safe, but latency is not fully bounded when a socket stays open without responding | The API returns `503`; the timeout limitation remains documented |

Operationally, this is still a local shadow-mode build: endpoints are unauthenticated, live feature state is per process, review items are not resolvable through a workflow, and the coordination graph is a shared-token projection rather than a second campaign classifier.

## Verification

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=riskloom --cov-report=term-missing --cov-fail-under=90
uv run alembic check
```

Integration tests use disposable PostgreSQL and fake every Razorpay and Gemini call. The suite enforces at least 90% branch-aware coverage.

<details>
<summary>Locked artifact hashes</summary>

| Artifact | SHA-256 |
| --- | --- |
| `model.json` | `3db8dafef643261c0df559cab632cfaf6fc45be54f38c1f4a621ef5af84039d4` |
| `training_report.json` | `4ff96556f1df49d7c44c29703c328753a6ea0ef197410976100e84751943ea6d` |
| `manifest.json` | `00aa16380eee1dcfa26fe9c89ed0eb8f866e75e98bd7a7ba89f9cc228c792f2e` |
| `evaluation.json` | `11251cef0dade5d14d2d1a85fe3822126e01c2a354494dd09b720c679244c40d` |
| feature `manifest.json` | `15337d0e9220f7ca96b4ded8157bc3ad29f38a6c5db9d357dea00b09371f28ba` |

These five values are pinned in `src/riskloom/runtime_bundle.py`, so the installer refuses any
archive whose contents differ, including one whose own manifest has been re-signed to match a
tampered payload.

The whole archive is pinned separately, and checked before any parsing at all, so a reshaped or
tampered file is rejected by a hash comparison rather than by format-parsing code. The archive does
not record its own outer hash — that value lives only in source, because a container that declares
the value it is judged by proves nothing. The five member hashes remain independently authoritative:
both layers must hold.

</details>

## Technology

- **Serving:** FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL
- **Modeling:** NumPy, scikit-learn, strict local JSON inference
- **Payments:** Razorpay test-mode orders and verified webhooks
- **Explanations:** Gemini behind a non-authoritative enrichment boundary
- **Operations:** Docker Compose, `uv`, Ruff, mypy, pytest

## Further reading

- [`docs/JUDGING.md`](docs/JUDGING.md) is the shortest rubric-to-evidence map for evaluators.
- [`submission_manifest.json`](submission_manifest.json) exposes the same verified claims as strict JSON.
- [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) contains the complete engineering, evaluation, and failure record.
- [`AGENTS.md`](AGENTS.md) defines the repository's durable safety and implementation rules.
- [`.env.example`](.env.example) documents the local configuration surface.
- [Razorpay AI Buildathon](https://razorpay.com/buildathon) contains the Track 02 brief.

## License

RiskLoom is available under the [MIT License](LICENSE).
