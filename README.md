# RiskLoom

RiskLoom is a defense-only, shadow-mode risk-management backend for detecting coordinated
card-testing activity. Day 1 establishes secure Razorpay test-mode webhook ingestion and an
internal Orders API adapter. It does not score, block, capture, or otherwise decide payments.

Day 2 adds an isolated deterministic simulator for privacy-safe checkout-attempt domain events
and an in-process event replayer. It does not send requests, execute payments, persist simulation
data to PostgreSQL, or expose another API route.

The generated artifacts are synthetic evaluation fixtures, not observations of real fraud or
payment traffic. RiskLoom makes no production fraud-accuracy claim from these datasets.

## Day 1 capabilities

- Python 3.11 FastAPI modular monolith.
- PostgreSQL persistence through SQLAlchemy 2 and Alembic.
- Exact-raw-body Razorpay webhook HMAC-SHA256 verification.
- Database-enforced idempotency using `X-Razorpay-Event-Id`.
- Append-only payment observations that tolerate out-of-order delivery.
- Allowlisted audit projections and exact-body SHA-256 digests; raw bodies are never persisted.
- Internal-only Razorpay test-mode Orders client using httpx.
- Structured logs, liveness/readiness checks, and PostgreSQL integration tests.

## Day 2 deterministic simulation

The `riskloom.simulation` package produces four canonical UTF-8/LF artifacts. JSON is compact and
key-sorted, identifiers are UUIDv5 values serialized through `uuid.hex`, and rates are integer
ratios plus fixed-decimal strings. Given Python 3.11, the same seed, generator/config versions, and
effective configuration, every artifact is byte-identical regardless of the output directory.

- `events.jsonl` contains strict model-visible synthetic checkout-attempt events only.
- `labels.jsonl` contains the separately typed evaluation truth joined one-to-one by `event_id`.
- `report.json` contains sorted aggregate counts, ratios, and reuse summaries without raw IDs.
- `manifest.json` contains versions, effective configuration, dataset identity, split boundaries,
  and SHA-256 metadata for the other three files. It intentionally does not hash itself.

Only `event_id` is globally unique. Merchant, checkout, customer, device, network, session, and
payment-instrument tokens are deliberately reusable so the data can represent legitimate retries,
flash sales, shared infrastructure, ordinary failures, and harmless coordinated-risk patterns.
The test split uses a controlled entity-reuse shift that exists only in test labels and generated
entity relationships, never as a model-visible marker.

That shift has one fixed direction and deterministic policy. For attacks, the test split must have
at least twice the unique-device-per-attack-event ratio and twice the
unique-session-per-attack-event ratio of both train and calibration. Validation uses integer
cross-products, so the reciprocal attack-events-per-device and attack-events-per-session ratios
must decrease without floating-point comparisons. Network coordination remains concentrated in
every split: unique attack networks may be at most 5,000/10,000 of attack events and at least
9,000/10,000 attack events must retain a network token. These thresholds are part of the effective
configuration and report; payment instruments are not used to define the shift.

The built-in profiles allocate every scenario by exact integer event counts in every chronological
split; generation never rounds or samples scenario quotas.

| Profile | Train | Calibration | Test | Total | Timeline |
| --- | ---: | ---: | ---: | ---: | --- |
| Smoke | 1,400 | 300 | 300 | 2,000 | 6 UTC days (4/1/1) |
| Development | 66,000 | 17,000 | 17,000 | 100,000 | 30 UTC days (20/5/5) |

Each split is exactly 70% normal, 8% legitimate retry, 12% flash sale, 5% shared
infrastructure, 3% legitimate failure, and 2% attack-labelled campaign events. Retry chains and
campaigns consume exactly their assigned event quotas.

Day 2.1 preserves configuration and algorithm `1.0.0` byte-for-byte, including its UUID namespace,
PRNG draw order, campaign construction, canonical artifacts, and validation of historical
datasets. Schema `1.1.0` is used for the development profile's calibration placement. Its canonical
effective-configuration SHA-256 is recorded in the manifest and binds every UUID and independent
PRNG stream, so changing output-affecting configuration with the same seed changes the generated
identity and data. The tracked development contract rejects changes to its approved split quotas,
campaign profiles, controlled test shift, or placement settings before generation.

Schema `1.1.0` profile contracts are selected from the effective configuration, never from a file
or directory name. `development` is the exact locked 100,000-event contract. Generic `smoke`
fixtures are limited to 3,000 total events, 1,400 events and 28 attacks per split, 60 attacks total,
six total days and four days per split, and ten campaigns per split. Protected placement is further
limited to five campaigns per side and 4,096 sampling attempts per campaign. These are the smallest
ceilings that retain the tracked six-day smoke shape and the reduced 3,000-row irregular-placement
test profile while preventing a relabelled development-scale configuration. Loading, generation
preflight, and artifact validation all apply the same profile-contract validator. Algorithm `1.0.0`
retains its historical configuration behavior and bytes.

Every public generation path first dumps the supplied configuration in Python mode, reconstructs a
fresh strict `GeneratorConfig`, and uses only that isolated snapshot. This occurs before output-path
inspection or staging, so post-construction mutation, schema downgrade, unknown fields, and invalid
numeric types fail without changing the destination.

The ten development calibration campaigns contain exactly 34 attack events each. Five complete
campaigns occur before a protected timestamp and five occur at or after it; none crosses it. The
timestamp is computed with integer milliseconds as
`calibration_start + floor(calibration_duration_ms * 6000 / 10000)`. Campaign windows are sampled
from independent seeded streams across the full permitted intervals, with no equal periodic slots.
They cannot overlap and must retain at least a five-minute gap. These are construction controls for
synthetic evaluation data, not operational payment-testing instructions.

Generate the smoke profile with reserved synthetic data:

```powershell
uv run python -m riskloom.simulation generate `
  --config configs/simulation/smoke.json `
  --seed 20260820 `
  --output-dir artifacts/simulations/smoke
```

Generate the larger development artifact only as a deliberate manual check, not in pytest:

```powershell
uv run python -m riskloom.simulation generate `
  --config configs/simulation/development.json `
  --seed 20260820 `
  --output-dir artifacts/simulations/development-v1.1.0-config-bound-a
```

Generated datasets under `artifacts/simulations/` are ignored by Git. Generation refuses unsafe
root/repository output targets, non-empty destinations, symlinked destinations, and overwrite of
unknown files. `--overwrite` is accepted only for a directory containing the four recognized
RiskLoom artifacts whose existing manifest, hashes, canonical bytes, report, and schemas all
validate. Files are validated in a sibling staging directory, then published with the manifest
last. The manifest is a completeness marker; publication is not a fully atomic four-file
filesystem transaction, and interrupted or partial publication fails later validation.

Replay model-visible events locally with no delay:

```powershell
uv run python -m riskloom.simulation replay `
  --events artifacts/simulations/smoke/events.jsonl `
  --timing no-delay
```

Replay is in-process and event-only. It has no label argument or import, HTTP/socket/URL target,
Razorpay adapter, retry, or remote transport. Scaled timing is bounded and intended for tests with
an injected fake clock; do not wait through a real multi-day dataset timeline.

Simulation events cannot contain PAN, CVV/CVC, expiry, email, phone/contact, address, VPA/UPI ID,
IP address, raw payload, names, or arbitrary free-form fields. Strict schemas reject unknown keys,
and validation recursively checks the explicit prohibited-field denylist. Safe synthetic tokens
are typed and prefixed; `payment_instrument_token` is explicitly allowed. The generator has no
name field, name configuration, or name-generation dependency/code path. Labels may be inspected
only to validate construction and evaluate future work; Day 2 does not add model training.

## Day 3 causal temporal features

The isolated `riskloom.features` package converts the chronological model-visible Day 2 event
stream into exactly 75 integer-valued temporal, velocity, and coordination features. It does not
read labels, split boundaries, scenarios, campaigns, generator metadata, or the Day 2 manifest.
It does not expose an endpoint, persist features, contact a network, score risk, train a model, or
make a payment decision.

For an event at time `t`, a window of `W` seconds contains only already processed events satisfying
`t - W < prior_time <= t`. An event exactly `W` seconds old is expired. Previously processed events
at the same timestamp are included in deterministic event-ID order. Processing always validates
and evicts first, computes and validates the current feature record from prior state, then updates
state with the current event. The current outcome can therefore affect only future feature rows.

The version 1.0.0 schema contains:

| Family | Features |
| --- | ---: |
| Safe current-event amount/time/channel/missingness | 9 |
| Checkout retry history within 3,600 seconds | 3 |
| Merchant rolling history | 15 |
| Device rolling history | 12 |
| Network rolling history | 15 |
| Instrument rolling history | 10 |
| Session rolling history | 6 |
| Prior 300-second failure rates in integer basis points | 5 |
| **Total** | **75** |

Rolling windows are locked to 60, 300, and 3,600 seconds. Deques bound observation state by event
time, and reference-counted relationship counters preserve exact distinct counts during eviction.
Missing devices and networks never share a null bucket. Checkout history also expires after 3,600
seconds. Failure rates use floor integer arithmetic and return zero when no prior attempt exists.

Extract and validate smoke features:

```powershell
uv run python -m riskloom.features extract `
  --events artifacts/simulations/smoke/events.jsonl `
  --config configs/features/default.json `
  --output-dir artifacts/features/smoke

uv run python -m riskloom.features validate `
  --events artifacts/simulations/smoke/events.jsonl `
  --config configs/features/default.json `
  --input-dir artifacts/features/smoke
```

The output contains canonical `features.jsonl`, aggregate-only `report.json`, and `manifest.json`.
The manifest records the exact source-events SHA-256, effective feature configuration, versions,
row count, sizes, and hashes of the feature and report files. It does not hash itself or contain a
path, timestamp, label, seed, or split. Dataset identity depends on the exact source bytes and
canonical effective configuration, not the output directory.

Feature rows contain only `event_id`, `occurred_at`, and the exact 75-key integer feature mapping.
They never emit current outcome, failure category, currency, entity tokens, labels, truth,
campaign data, split values, scores, thresholds, decisions, or predictions. Reports contain only
integer aggregate distributions and aggregate state-size diagnostics.

Extraction and validation stream source events and feature rows without retaining full datasets in
memory. Exact percentile frequency counters retain integer frequencies only. Existing output can
be replaced with `--overwrite` only when it is already a fully valid RiskLoom feature dataset for
the same source bytes and configuration. Staging and manifest-last publication provide a
completeness marker, not a fully atomic three-file transaction. Generated features under
`artifacts/features/` are ignored and must not be committed. Generate the 100,000-row development
artifact only as an explicit manual verification command:

```powershell
uv run python -m riskloom.features extract `
  --events artifacts/simulations/development-v1.1.0-config-bound-a/events.jsonl `
  --config configs/features/default.json `
  --output-dir artifacts/features/development-v1.1.0-config-bound
```

## Day 4 offline model locking

The isolated `riskloom.modeling` package trains two fixed tabular candidates against the exact
approved development artifacts: standardized logistic regression and conservative gradient
boosting. NumPy and scikit-learn are runtime dependencies only for offline fitting and independent
validation; the locked model is canonical, data-only JSON and never pickle, cloudpickle, joblib, or
executable estimator state.

The modeling configuration pins every source identity and hash, all candidate and Platt-calibrator
parameters (including `random_state`), the fixed 1:25 false-positive/false-negative cost policy,
feature order, and the calibration boundary. The source simulation must be profile `development`,
generator/configuration 1.1.0, and the exact approved effective-configuration fingerprint. Labels
are validated through the Day 2 manifest and exact label-file hash; model training never opens raw
events.

Calibration is divided only by timestamp at
`calibration_start + floor(calibration_duration_ms * 6000 / 10000)`. Rows strictly before the
timestamp form `calibration_fit`; equal or later rows form `policy_selection`. The approved data
currently yields 9,048 and 7,952 rows respectively, with 170 attacks and five complete campaigns
on each side. Preprocessing, balanced weights, and both candidates fit on train only. Platt
calibration fits on calibration-fit only. Candidate and threshold selection use policy-selection
only. Average precision and trapezoidal PR-AUC are reported as distinct metrics.

Policy reports include row/class counts, prevalence, confusion counts, precision, recall,
specificity, F1, false positives per 10,000 legitimate events, review workload, configured cost,
ranking metrics, Brier score, log loss, and fixed ten-bin reliability/ECE results. Legitimate
scenario slices report false-positive counts, rates, per-10,000 rates, and abstract cost units;
empty slices use `null` rates. Campaign summaries retain completely missed campaigns as zero
flagged events and report campaign recall plus first-attack-to-first-flag delay in integer
milliseconds. These metrics describe synthetic data only and are not production accuracy or
merchant-loss claims.

Point the commands at directories whose manifests match the exact identities pinned in
`configs/modeling/default.json`. In the current local verification workspace, those approved bytes
are in the `development-v1.1.0-config-bound-a` simulation directory and
`development-v1.1.0-config-bound` feature directory:

```powershell
uv run python -m riskloom.modeling train `
  --simulation-dir artifacts/simulations/development-v1.1.0-config-bound-a `
  --feature-dir artifacts/features/development-v1.1.0-config-bound `
  --config configs/modeling/default.json `
  --output-dir artifacts/models/development

uv run python -m riskloom.modeling validate-model `
  --simulation-dir artifacts/simulations/development-v1.1.0-config-bound-a `
  --feature-dir artifacts/features/development-v1.1.0-config-bound `
  --config configs/modeling/default.json `
  --model-dir artifacts/models/development
```

Training discards held-out feature rows and never accesses or validates held-out targets.
`validate-model` first validates the locked canonical artifact, independently retrains from
train/calibration data, compares byte identity, and then may use a bounded deterministic sample of
held-out features only for sklearn/portable inference parity. Training reports contain aggregate
train, calibration-fit, and policy-selection results only—never raw identifiers, predictions,
held-out counts, or held-out metrics.

The `evaluate-test` command exists for a later, separately approved gate. It performs portable
inference against an already locked model and has no fit path. Do not run it on the official
development partition before that approval. Generated model and evaluation directories are ignored
by Git, publication is manifest-last with no overwrite option, and a non-empty destination is
always refused.

Byte determinism is guaranteed only for the same Python, NumPy, scikit-learn and dependency-lock
versions, exact source bytes, effective configuration, operating system, CPU/numerical environment,
and fixed seeds. It is not claimed across CPU architectures, BLAS implementations, platforms, or
dependency versions.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop with Linux containers

No local PostgreSQL client is required.

## Local setup

```powershell
Copy-Item .env.example .env
uv sync
docker compose up -d --wait postgres
uv run alembic upgrade head
uv run uvicorn riskloom.main:app --reload --port 8000
```

Replace the Razorpay placeholders in `.env` only when exercising the internal client or receiving
test-mode webhooks. Live-mode key IDs are rejected during configuration validation.

PostgreSQL binds to host port `5432` by default. If that port is already in use, change both local
values in the ignored `.env` file so Compose and the application continue to address the same port;
for example:

```dotenv
RISKLOOM_DATABASE_URL=postgresql+asyncpg://riskloom:riskloom_local_only@127.0.0.1:5433/riskloom
RISKLOOM_POSTGRES_PORT=5433
```

Keep `.env.example` at the documented `5432` default and use `.env` for machine-specific overrides.

Check the service:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

## API

- `GET /health/live` checks the process only.
- `GET /health/ready` checks PostgreSQL connectivity.
- `POST /api/v1/webhooks/razorpay` accepts signed Razorpay webhooks.

There is deliberately no public order-creation endpoint. The Orders client is an internal adapter.

## Webhook security and privacy

The webhook endpoint streams the request body into a bounded in-memory buffer. It verifies
`X-Razorpay-Signature` against those exact bytes before JSON parsing. The bytes are then hashed and
discarded when the request completes. They are never written to PostgreSQL, logs, temporary files,
exceptions, or error reports.

Only a conservative allowlist of documented event and payment fields is retained. Unknown keys,
free-form values, contact data, cardholder data, network addresses, card credentials, VPA data,
notes, descriptions, token data, and acquirer data are excluded. Fixtures use reserved synthetic
values only.

## Migrations and tests

```powershell
uv run alembic upgrade head
uv run alembic downgrade base
uv run alembic upgrade head
uv run alembic check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=riskloom --cov-report=term-missing --cov-fail-under=90
```

Integration tests start an isolated PostgreSQL 16 container and never call Razorpay. Docker must be
running. To stop the development database without deleting its volume:

```powershell
docker compose down
```

## Configuration

All application settings use the `RISKLOOM_` prefix. Secrets are represented as Pydantic
`SecretStr` values and must not be interpolated into logs or exceptions. See `.env.example` for the
complete local configuration surface.
