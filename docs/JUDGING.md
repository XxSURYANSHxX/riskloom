# RiskLoom evaluator guide

## Thirty-second verdict

RiskLoom addresses a specific merchant-loss pattern: low-value card-testing attempts can look
legitimate individually while forming a coordinated abuse ring across devices, sessions, networks,
instruments, merchants, and time. It is a defense-only Track 02 risk manager that computes 75
causal temporal features, applies one locked calibrated threshold, records the result in an
append-only ledger, and exposes the evidence through a read-only operations dashboard.

On the one locked chronological synthetic test split, the model detected all 3 campaigns and 332
of 340 attack events across 17,000 events. Precision was 76.85%, the false-positive rate was 0.60%
(100 false positives and 16,560 true negatives), average precision was 96.40%, and ROC-AUC was
98.86%. These are offline synthetic-data measurements, not live-serving or production-accuracy
claims. The authoritative values are in the installed
[`evaluation.json`](../artifacts/evaluations/development/evaluation.json), whose hash is pinned in
[`runtime_bundle.py`](../src/riskloom/runtime_bundle.py); the complete evaluation narrative is in
the [Day 4 build log](BUILD_LOG.md#day-4-offline-model-locking).

## Razorpay rubric-to-evidence map

| Criterion | RiskLoom answer | Exact evidence |
| --- | --- | --- |
| Problem taste | Detect coordinated card testing before individually ordinary attempts compound into merchant loss. | [README problem statement](../README.md#why-riskloom); deterministic legitimate and attack scenarios in [`configs/simulation/development.json`](../configs/simulation/development.json); [simulation design record](BUILD_LOG.md#day-2-deterministic-simulation) |
| Build quality | A typed FastAPI modular monolith with PostgreSQL, append-only audit records, causal state, portable JSON inference, deterministic graph layout, pinned runtime artifacts, and fail-closed startup. | [Architecture](ARCHITECTURE.md); [`FeatureEngine`](../src/riskloom/features/engine.py); [`preflight.py`](../src/riskloom/services/preflight.py); [`runtime_bundle.py`](../src/riskloom/runtime_bundle.py); [current verification record](BUILD_LOG.md#current-verified-state) |
| AI judgment | The locked calibrated model and its single threshold make risk decisions. Gemini is lazy post-decision enrichment for a final `DENY`; its package is isolated from scoring and persistence. | [`decisions.py`](../src/riskloom/serving/decisions.py); [`explanations`](../src/riskloom/explanations); [`test_isolation.py`](../tests/unit/explanations/test_isolation.py); [Day 8 record](BUILD_LOG.md#day-8-llm-generated-incident-explanations) |
| Failure recovery | Runtime-bundle publication originally left a half-installed tree when a file publication failed. The installer now performs ownership-aware scoped rollback and proves a clean retry. | [Gate I0.1 incident](BUILD_LOG.md#security-review-gate-i01); [`test_runtime_bundle.py`](../tests/unit/test_runtime_bundle.py), especially `test_a_publication_failure_rolls_the_destination_back_exactly`, `test_rollback_never_removes_a_pre_existing_identical_file`, and `test_installation_after_a_rolled_back_failure_succeeds` |
| Track 02: honest precision, recall, and false-positive cost | 332/340 event recall, 332/432 precision, 100/16,660 FPR, and 3/3 campaign recall. The 100 false positives cost 100 abstract units under the locked 1:25 FP/FN weights; no currency value is claimed. | Installed [`evaluation.json`](../artifacts/evaluations/development/evaluation.json); [`metrics.py`](../src/riskloom/modeling/metrics.py); [held-out report](BUILD_LOG.md#day-4-offline-model-locking); [`submission_manifest.json`](../submission_manifest.json) |

The repository has no GitHub Actions workflow. Verification is local and explicit through the
commands below; the current recorded results are in the
[build log](BUILD_LOG.md#current-verified-state).

## Held-out evaluation and confusion counts

The model, calibration, feature order, threshold, and cost weights were frozen before the
chronologically later test period was opened. The evaluation command uses portable inference from
the locked JSON model and has no fit path.

| Quantity | Numerator / denominator | Value |
| --- | ---: | ---: |
| Test events | — | 17,000 |
| Attack events | — | 340 |
| Campaign recall | 3 / 3 | 100% |
| Event recall | 332 / 340 | 97.65% |
| Precision | 332 / (332 + 100) | 76.85% |
| False-positive rate | 100 / (100 + 16,560) | 0.60% |
| Average precision | — | 96.40% |
| ROC-AUC | — | 98.86% |

| Actual class | Above locked threshold | Below locked threshold |
| --- | ---: | ---: |
| Attack | 332 true positives | 8 false negatives |
| Legitimate | 100 false positives | 16,560 true negatives |

The exact floating-point values, reliability bins, hard-negative slices, campaign delay, model ID,
and evaluation ID remain in [`evaluation.json`](../artifacts/evaluations/development/evaluation.json).
Its SHA-256 is
`11251cef0dade5d14d2d1a85fe3822126e01c2a354494dd09b720c679244c40d`.

## False-positive operational cost

The locked evaluation cost is:

```text
total_cost = false_positives * 1 + false_negatives * 25
           = 100 * 1 + 8 * 25
           = 300 abstract units
```

The false-positive component is therefore 100 abstract units. These weights make errors
comparable during threshold selection; they are not rupees, savings, interchange fees, chargeback
amounts, or estimates of merchant loss.

The evaluation artifact records a binary threshold result, not executed payment actions. In the
live decision code, a comparable event at or above the threshold maps to `DENY` and creates no
Razorpay test-mode order. `REVIEW` is different: it is reached only when feature computation,
scoring, order creation, or the process order budget prevents safe completion, and it creates a
pending review item. The held-out confusion matrix did not run those operational failure paths, so
it must not be read as 100 review items or as evidence that 100 real payments were affected. See
[`decisions.py`](../src/riskloom/serving/decisions.py),
[`preflight.py`](../src/riskloom/services/preflight.py), and the
[Day 6 decision flow](BUILD_LOG.md#day-6-live-checkout-preflight-scoring).

## AI judgment: where AI is and is not used

RiskLoom has two deliberately separate AI surfaces:

1. The core risk decision uses the locked gradient-boosting model, Platt calibrator, 75 causal
   features, and one full-float64 threshold from `model.json`. A score below the threshold is
   `ALLOW`; a score at or above it is `DENY`. No LLM and no second risk threshold is reachable.
2. Gemini may generate a structured explanation only after a `DENY` is final and a human requests
   it. The input contains numbers, booleans, and closed enums rather than free text or tokens. The
   output is validated and sanitised, and cannot change the score, threshold, risk decision,
   action, ledger row, or payment behavior.

`REVIEW` is an operational fail-safe action, not a model band. The package-isolation tests enforce
both directions of the boundary: the decision path cannot load explanations, and explanations
cannot load serving, database, features, modeling, policy, or preflight code. Evidence:
[`serving/decisions.py`](../src/riskloom/serving/decisions.py),
[`explanations`](../src/riskloom/explanations),
[`test_isolation.py`](../tests/unit/explanations/test_isolation.py), and
[`test_train_serve_parity.py`](../tests/unit/serving/test_train_serve_parity.py).

## Architecture and decision flow

```text
synthetic or test-mode checkout request
                 |
                 v
claim event id in PostgreSQL before touching causal state
                 |
                 v
server timestamp + 75-feature compute-before-update engine
                 |
                 v
locked JSON model -> Platt probability -> one threshold
                 |
        +--------+--------+
        |                 |
      ALLOW              DENY
 capped test order      no order
        |                 |
        +--------+--------+
                 v
      append-only final decision ledger
                 |
        +--------+----------------+
        v                         v
 read-only dashboard       optional requested Gemini
 and coordination graph    explanation for final DENY
```

If a decision cannot safely complete, the action is `REVIEW`, the reason is recorded, and no order
is created. Architecture and safety boundaries are exercised in
[`test_checkout_preflight.py`](../tests/integration/test_checkout_preflight.py), while exact
offline/online feature parity is exercised in
[`test_train_serve_parity.py`](../tests/unit/serving/test_train_serve_parity.py). The dashboard
screenshots in the [README walkthrough](../README.md#product-walkthrough) show stored decisions,
pseudonymous shared-signal topology, and offline aggregates; they are demonstrations, while the
linked reports and tests are the evidence.

## Failure recovery

The strongest documented development recovery is the Gate I0.1 runtime-installer rollback defect.
It is separate from the deliberate Day 9 service failure drill.

| Question | Evidence-backed answer |
| --- | --- |
| Symptom | Injecting a failure while publishing the third of five runtime files left the first two installed; the next attempt encountered a half-populated artifact tree. |
| Root cause | Publication staged and replaced files sequentially but had no record of which destination files the current attempt created and no failure cleanup. |
| Why it mattered | The installer distributes the exact model, manifests, and held-out aggregates required by startup. Partial state undermined the completeness boundary and made retry behavior unreliable. |
| Fix | Classify every destination before writing, refuse conflicts up front, and on failure remove exactly the files created by that attempt. Preserve pre-existing identical files and never claim multi-directory transactionality. |
| Regression prevention | Parameterized failure injection covers publication steps 1 through 5; separate tests preserve a pre-existing identical file and prove a subsequent installation succeeds. |
| Final verification | The focused tests are in [`test_runtime_bundle.py`](../tests/unit/test_runtime_bundle.py); the incident and review outcome are recorded under [Gate I0.1](BUILD_LOG.md#security-review-gate-i01). |

## Reproduction path

### Fast evaluator preflight

Prerequisites are Python 3.11 and `uv`. The pinned runtime bundle must already be installed; on a
fresh clone, install it with the fixed-source downloader first:

```powershell
uv sync --frozen
uv run python scripts/runtime_bundle.py install
uv run python scripts/preflight_check.py
```

The preflight command is read-only and normally completes in seconds; this evaluator-readiness run
measured 5.9 seconds including `uv` startup. It does **not** require Docker, a running PostgreSQL
server, Razorpay credentials, or Gemini credentials. It checks the two tracked configurations, all
four startup artifacts against source-pinned SHA-256 hashes, reports the optional held-out
evaluation separately, and invokes the same serving-bundle binding used at application startup.

Success exits 0 and ends with:

```text
All startup requirements present and the serving binding succeeds.
```

A missing or invalid startup input exits 1, names every blocking path and state, and points to the
fixed `runtime_bundle.py install` command. This preflight establishes artifact and serving binding;
it does not claim database readiness, migration correctness, or endpoint health.

Gemini is optional. Without `RISKLOOM_GEMINI_API_KEY`, scoring, threshold decisions, the ledger,
dashboard, drift diagnostics, and deterministic fail-safe reason rendering remain available; only
generated narrative enrichment reports unconfigured. Razorpay placeholders are sufficient to start,
while an actual `ALLOW` order requires test-mode credentials.

### Complete validation

Docker is required for the complete pytest run because integration tests create disposable
PostgreSQL. Razorpay and Gemini calls are faked. Run:

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=riskloom --cov-report=term-missing --cov-fail-under=90
uv run alembic check
```

For a serving demo after preflight, copy `.env.example` to `.env` and run
`docker compose up --build -d`; see the [README run path](../README.md#run-locally) and the verified
[clean-clone drill](BUILD_LOG.md#gate-i1-public-release-clean-clone-drill).

## Safety boundaries

- Defense-only, synthetic-data, local shadow-mode scope.
- Only Razorpay key IDs beginning with `rzp_test_` are accepted.
- No capture, refund, settlement, payment modification, or public order-creation endpoint.
- Webhook signatures cover exact raw request bytes before JSON parsing; raw bytes are not stored.
- The ledger stores an explicit pseudonymous projection, not raw payment data or free-form fields.
- Chronological train/calibration/policy-selection/test isolation is strict.
- Gemini is post-decision, optional, allowlisted, bounded, and unable to affect the decision path.
- Adversarial analysis is local and offline and accepts no external target.

The durable implementation rules are in [`AGENTS.md`](../AGENTS.md), with the public summary in
[Safety by design](../README.md#safety-by-design).

## Known limitations

The canonical consolidated list is [What testing revealed](../README.md#what-testing-revealed).
The most decision-relevant boundaries are:

- Held-out metrics describe offline scoring with true prior outcomes on one chronological synthetic
  split. They are not measured live-serving or production accuracy.
- Under the measured live authorized-outcome assumption, recall fell from 0.856 to 0.600 and cost
  rose from 763 to 2,279 abstract units because 18 failure-derived features read low.
- Boundary-spaced synthetic attacks reduced recall to 0.00-0.12; exact 3,600-second spacing detected
  0/120 attack events. The locked model was disclosed rather than silently changed.
- The locked threshold is one float64 step above an eight-row probability tie cluster. The
  diagnostic is non-fatal, and published held-out metrics use portable inference.
- Live temporal state is per process and is cold after restart. Review items have no resolution
  workflow, endpoints are unauthenticated, and a frozen database can remain slow until the
  connection dies even though the path fails closed.

Mechanisms and full qualifications are in the [Day 6 blind-spot record](BUILD_LOG.md#known-limitation-live-serving-accuracy-is-not-measured-to-held-out-standard),
[Gate H0](BUILD_LOG.md#gate-h0-adversarial-stress-test), and
[Day 9 failure record](BUILD_LOG.md#what-the-storage-drill-actually-found).

## Evidence index

| Evidence | What it establishes |
| --- | --- |
| [`submission_manifest.json`](../submission_manifest.json) | Strict machine-readable project, release, metric, safety, reproduction, and evidence metadata |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | End-to-end runtime, audit, offline, trust, external-service, and failure boundaries |
| [`evaluation.json`](../artifacts/evaluations/development/evaluation.json) | Authoritative held-out aggregates and confusion counts; installed from the pinned runtime bundle |
| [`model.json`](../artifacts/models/development/model.json) and [`manifest.json`](../artifacts/models/development/manifest.json) | Locked data-only model, threshold, identities, and source contract |
| [`runtime_bundle.py`](../src/riskloom/runtime_bundle.py) | Immutable runtime tag, archive hash, five artifact hashes, validation, and safe installation |
| [`preflight_check.py`](../scripts/preflight_check.py) | Short read-only startup-input and serving-binding check |
| [`README.md`](../README.md) | Judge card, product behavior, screenshots with textual descriptions, architecture, run path, safety, and consolidated limitations |
| [`BUILD_LOG.md`](BUILD_LOG.md) | Chronological engineering decisions, measured failures, evaluation record, and exact verification history |
| [`test_submission_manifest.py`](../tests/unit/test_submission_manifest.py) | Strict JSON, evidence path, artifact agreement, arithmetic, hash, and safe-claim validation |
| [`test_runtime_bundle.py`](../tests/unit/test_runtime_bundle.py) | Runtime distribution, rollback, download, archive, hash, and preflight regression coverage |
| [`test_checkout_preflight.py`](../tests/integration/test_checkout_preflight.py) | PostgreSQL-backed ALLOW/REVIEW/DENY, idempotency, concurrency, budget, and ledger behavior |
| [`test_train_serve_parity.py`](../tests/unit/serving/test_train_serve_parity.py) | Exact all-75-feature train/serve parity through the real extraction and online host |
| [`test_isolation.py`](../tests/unit/explanations/test_isolation.py) | Bidirectional decision/explanation isolation at source and import time |
| [`compose.yaml`](../compose.yaml) and [`Dockerfile`](../Dockerfile) | One-shot migration, read-only artifact mount, non-root image, and runtime secret boundary |

Release integrity: `v1.0.3-submission` at
`ff36ba091f5bcf44b45e40044139996663f03bca` is the immutable engineering and runtime baseline
immediately preceding this evaluator-readiness package. The package changes documentation,
machine-readable metadata, and validation tests only and is suitable for a new source submission
release; it does not change the model, runtime bundle, artifact hashes, features, APIs, schema,
dependencies, or inference behavior. `v1.0.3-runtime` remains the separate immutable runtime-asset
release, and existing tags remain immutable.
