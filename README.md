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

The `evaluate-test` command performs portable inference against an already locked model and has no
fit path. The official development held-out evaluation has now been run exactly once against the
locked `development` model. Over 17,000 held-out events at 2.0% attack prevalence, it reaches 97.6%
recall and 76.9% precision at the locked threshold, with average precision 0.964. All three
held-out campaigns were detected; that is a 3-of-3 denominator on a single synthetic split, not a
general campaign-detection claim.

Known limitation: two legitimate traffic patterns account for 90% of all false positives —
`shared_infrastructure` (6.9% false-positive rate) and `legitimate_retry` (2.3%) — and the model is
overconfident in the mid probability range, where the 0.4-0.8 reliability bins predict far higher
risk than the observed attack rate in them. These figures describe synthetic data only and are not
production accuracy or merchant-loss claims.

Generated model and evaluation directories are ignored by Git, publication is manifest-last with no
overwrite option, and a non-empty destination is always refused.

Byte determinism is guaranteed only for the same Python, NumPy, scikit-learn and dependency-lock
versions, exact source bytes, effective configuration, operating system, CPU/numerical environment,
and fixed seeds. It is not claimed across CPU architectures, BLAS implementations, platforms, or
dependency versions.

### Decision-boundary diagnostic

Parity between the fitted estimator and the portable JSON model is checked on probabilities, within
`parity_absolute_tolerance`. That does not guarantee the two agree on the discrete decision. A
gradient-boosting model with shallow trees can legitimately produce very few distinct calibrated
probabilities on a skewed feature set -- the locked development model produces 15 distinct values
across 7,952 `policy_selection` rows -- so probabilities arrive in large ties, and a threshold that
lands on a tie-cluster boundary can flip every row in that cluster at once on a difference far
smaller than the tolerance.

The locked development threshold sits one unit in the last place above a cluster of eight
`policy_selection` rows (one attack, seven legitimate). Portable inference allows those eight rows,
while `training_report.json` recorded them as denied, so that report's `policy_selection` confusion
matrix and portable-model behaviour differ by those eight rows.

This is an internal-consistency gap only. **No published held-out evaluation number is affected.**
Every figure in the Day 4 section above was computed through portable inference on the held-out
partition and correctly describes deployed behaviour.

Rather than retroactively re-locking an already-accepted and already-published artifact, the
condition is now monitored. `validate-model` reports a non-fatal `decision_boundary` block naming
whether the threshold is an observed probability, the nearest observed value below it, and the
attack and legitimate row counts tied there. It is a diagnostic, never a pass or fail gate, and it
exists so any future re-lock or retraining run surfaces the same condition before it is accepted.

## Day 5 cost-aware policy band

The isolated `riskloom.policy` package replaces Day 4's single ALLOW/DENY threshold with a
three-tier ALLOW / REVIEW / DENY band. It never imports the simulation label module, directly or
transitively, and its routing function takes a calibrated probability and a band, nothing else.
Label-bearing orchestration lives in `riskloom.modeling.policy_ops`, where labels only ever score a
decision that has already been made.

Total cost extends the Day 4 function rather than replacing it:

```
total_cost = FN * false_negative_cost_units    (25, inherited and unchanged)
           + FP * false_positive_cost_units    (1,  inherited and unchanged)
           + N_review * review_cost_units      (3,  new abstract policy weight)
```

`N_review` counts every reviewed event regardless of its true label, because review is an
operational cost incurred whether or not the transaction turns out to be fraud. All three weights
are abstract units and none is a rupee-denominated claim. Both thresholds are fitted on
`policy_selection` by a deterministic sweep over observed probabilities that always includes the
incumbent threshold, so the band can always express the existing policy exactly.

Counterfactual validation uses a separate `policy-validation` profile with its own locked contract,
a seed used by no prior gate, and a window running 2026-03-01 to 2026-03-12 -- entirely after the
development window that ends 2026-01-31. The loader refuses any batch whose dataset ids, artifact
hashes, or configuration fingerprint match the locked development contract.

```powershell
uv run python -m riskloom.simulation generate `
  --config configs/simulation/policy-validation.json `
  --seed 20260905 `
  --output-dir artifacts/simulations/policy-validation

uv run python -m riskloom.features extract `
  --events artifacts/simulations/policy-validation/events.jsonl `
  --config configs/features/default.json `
  --output-dir artifacts/features/policy-validation

uv run python -m riskloom.modeling fit-policy-band `
  --simulation-dir artifacts/simulations/development-v1.1.0-config-bound-a `
  --feature-dir artifacts/features/development-v1.1.0-config-bound `
  --config configs/modeling/default.json `
  --model-dir artifacts/models/development `
  --policy-config configs/policy/default.json `
  --output-dir artifacts/policy/bands/development

uv run python -m riskloom.modeling validate-policy `
  --band-dir artifacts/policy/bands/development `
  --validation-simulation-dir artifacts/simulations/policy-validation `
  --validation-feature-dir artifacts/features/policy-validation `
  --config configs/modeling/default.json `
  --model-dir artifacts/models/development `
  --policy-config configs/policy/default.json `
  --output-dir artifacts/policy/comparisons/development
```

The banded policy never auto-activates. Approval requires the explicit `--approve` flag and is
refused whenever the band fails to beat the incumbent cost, exceeds the configured
false-positive-rate ceiling (default 100 basis points, configurable per merchant), or the
validation batch falls below the configured minimum of 2,000 rows and 100 attacks.

On the current fresh validation batch the fitted band beat the incumbent on cost -- 741 against 763
abstract units across 9,000 events -- but was refused approval because its false-positive rate of
132 basis points exceeds the 100 basis point ceiling. The incumbent policy also exceeds that
ceiling on the same batch, at 128 basis points. The fitted band additionally has an empty review
tier: with a false positive costing 1 unit and a review costing 3, denying is always cheaper than
reviewing, so a review tier can never be cost-optimal at these inherited weights. Making the review
tier reachable would require rescaling the cost units, which is a deliberate future decision rather
than something this gate tunes for.

## Day 6 live checkout-preflight scoring

`POST /api/v1/checkout/preflight` scores a single live checkout attempt and acts on it. This is a
deliberate expansion of the API surface, which previously held at two health routes and the webhook.

The decision uses the locked Day 4 model and its single `decision_threshold` only. Neither Gate C1
policy band takes part in any real decision, and an isolation test proves `riskloom.policy` is not
reachable from the live path even transitively.

```
probability >= decision_threshold  ->  DENY   (no order created)
otherwise                          ->  ALLOW  (create a Razorpay test-mode order)
```

REVIEW is an operational fail-safe tier rather than a risk band. It is reached only when a decision
cannot be safely completed -- feature computation failed, scoring failed, order creation failed, or
the process order budget is exhausted -- and the ledger records the underlying `risk_decision`
separately from the `action` so the downgrade stays auditable.

The order budget (`razorpay_max_orders_per_process`, default 5) counts order-creation *attempts*
rather than successes: it is reserved before the upstream call, so an attempt Razorpay rejects
still consumes one unit. That is the safer direction, since what needs bounding is outbound calls
to a payment provider, but it does mean the counter can exceed the number of orders that actually
exist.

Live features come from the unmodified Day 3 `FeatureEngine`, held warm in one process-wide
instance behind a single lock. The lock covers server-side timestamp assignment and the engine call
only; order creation and database writes happen outside it. State is **in-memory only and does not
survive a restart** -- after a restart the engine is cold and history features read zero until live
traffic rebuilds them. At startup the service refuses to run unless the running feature
configuration matches the effective configuration in the locked feature manifest, that manifest's
hash matches the one pinned in the modeling configuration, and the model's feature order matches
the live schema.

Every decision is recorded in the append-only `risk_decisions` ledger with pseudonymous fields
only; REVIEW additionally creates a `review_items` row. There is no auto-resolution of review items.

```powershell
uv run alembic upgrade head
uv run uvicorn riskloom.main:app --host 127.0.0.1 --port 8000
```

### Known limitation: live-serving accuracy is not measured to held-out standard

At preflight an attempt's outcome does not exist yet, so the online adapter advances feature state
with every attempt recorded as authorized. The 57 outcome-independent features are identical
between training and serving; the 18 failure-derived features read low for live traffic.

This has been measured rather than assumed. Replaying the 9,000-event policy-validation batch
through the locked model under both assumptions -- true outcomes, as offline training saw them,
versus every state-advancing outcome forced to authorized, as live serving assumes -- gives:

| Metric | True outcomes | Assumed authorized | Delta |
| --- | ---: | ---: | ---: |
| Recall | 0.8556 | 0.6000 | -0.2556 |
| Precision | 0.5768 | 0.1840 | -0.3928 |
| False-positive rate | 1.28% | 5.43% | +4.15 pp |
| Average precision | 0.4384 | 0.5746 | +0.1362 |
| ROC-AUC | 0.9260 | 0.8194 | -0.1066 |
| Total cost (FN*25 + FP*1) | 763 | 2,279 | +1,516 (+199%) |
| Attacks missed | 26 | 72 | +46 |

The degradation is material: under the live assumption the same locked model misses 2.8x as many
attacks and costs roughly three times as much. Average precision rises while every threshold-based
metric falls, because the blind spot shifts the score distribution rather than uniformly worsening
the ranking, leaving the fixed locked threshold badly placed for that distribution.

**Gate B2's held-out figures describe offline scoring with true outcomes and must not be quoted as
live-serving accuracy.** Webhook-driven failure reconciliation, which would close this gap, is named
future work and is not in this gate. The measurement is reproducible through
`riskloom.analysis.blindspot`.

## Day 7 risk-operations dashboard

A read-only operations view over data the earlier gates already write. It adds no computation, no
mutation path and no trust boundary: every figure it shows was decided and stored elsewhere, and
the dashboard only projects it. Five GET endpoints under `/api/v1/dashboard` serve JSON, and a
static client is mounted at `/dashboard`.

```powershell
uv run alembic upgrade head
uv run uvicorn riskloom.main:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000/dashboard
```

The client is hand-written ES modules with no build step and no framework, and it deliberately
holds no logic worth testing. Grouping, sizing, formatting and graph layout are computed in Python
and covered by pytest; the client draws what it is given. The one number it contributes is the
measured pixel size of the graph panel, which it forwards as `canvas_width` and `canvas_height`
because the server cannot otherwise know how wide the viewport is.

| View | Shows |
| --- | --- |
| Stream | Newest decisions first, one line each, refreshed every 3s. Probabilities render at full precision -- the locked threshold carries nineteen significant decimals and a real tie-cluster sits one ULP below it, so rounding for display would erase the distinction the system is built around. |
| Coordination | The shared-token graph: entities are hubs, decisions are the small nodes attached to them. Refreshed every 5s. |
| Case detail | One decision's stored fields beside ledger-derived co-occurrence context. |
| Ledger | The full filterable history, refreshed every 10s. |

Updates are polled rather than pushed. Events only appear when a checkout attempt is posted, so a
3-second poll already outpaces the event rate; SSE or WebSockets would add connection lifecycle and
a second server code path for no gain at this scale, and remain the documented upgrade route.

### The coordination graph is not campaign detection

The two panels on the coordination view come from different places and mean different things. This
distinction is load-bearing and the interface labels it in both directions.

The **graph** is live and derived entirely from the ledger. A hub appears when a stored device,
network or instrument token is shared by more than one decision; hub radius grows with the number
of attached decisions and ring thickness with the number of distinct token kinds that co-occur.
That is a projection of stored pseudonymous tokens, nothing more. **RiskLoom has no live campaign
detection** -- no model, threshold or classifier decides that these decisions form a campaign, and
the ledger has no campaign column. Shared tokens are evidence an operator interprets.

The **side panel** is offline and static. Its campaign figures are read from
`artifacts/evaluations/development/evaluation.json`, the Gate B2 held-out evaluation, which was
computed against ground-truth simulation labels that do not exist for live traffic. Those numbers
describe how the locked model performed on a historical labelled dataset. They are never recomputed
from live data, never updated by traffic, and must not be read as a claim about the events drawn to
their left.

Ledger co-occurrence counts on the case-detail view are likewise structural context, not model
features. The 75-feature vector is not stored, and recomputing it would be wrong -- the engine's
rolling state has moved on, so a recomputed vector would differ from the one the decision used. No
feature name is displayed anywhere in the dashboard.

### Known limitations

- **No authentication.** Every dashboard endpoint is unauthenticated and the client ships no login.
  This is acceptable only because the service is a local single-process build; the dashboard must
  not be exposed on a network interface as it stands.
- **No mutation path.** The dashboard is read-only in this gate. Review items surface only as
  counts -- a pending total on the summary and a per-decision count on case detail -- and cannot be
  listed, approved, resolved or overridden. No endpoint accepts anything but `GET`; every other verb
  answers 405 at the router, which is enforcement by routing rather than convention. Making review
  actionable is the natural next step, but it would pull the dashboard inside the safety boundary
  Days 4-6 were held to, and that needs its own gate.
- **`GET /api/v1/dashboard/model` returns 404 when the evaluation artifact is absent.** The
  `artifacts/` tree is Git-ignored, so a fresh clone has no `evaluation.json` and this is an
  ordinary state rather than an error. The endpoint 404s, the panel renders an explicit unavailable
  state, and startup is unaffected.
- **The dashboard reads the database the live path writes.** Queries are strictly read-only and no
  transaction outlasts a single statement, but the two share one PostgreSQL instance.

## Day 8 LLM-generated incident explanations

A denied checkout can now carry a short natural-language explanation, generated by Gemini and
grounded strictly in facts that were already computed and stored. The governing rule, carried from
every prior gate: **the LLM explains, it never decides.** It has no path to `risk_decision`,
`action`, `calibrated_probability`, `decision_threshold` or any other value in `risk_decisions`,
and that is enforced structurally rather than promised.

Generation is **lazy**: it happens when an operator asks for it from the case-detail view, never
automatically. Eager generation was rejected on a structural ground rather than a preference.
Producing prose inside the preflight path would require `riskloom.services.preflight` to import the
explanation module, which this gate's isolation rule forbids; the two cannot both hold. It also
keeps an external dependency out of the path that creates real Razorpay orders.

### Only a finalised DENY is eligible

REVIEW is never a risk band in this system. Its four causes are operational fail-safes, and a
REVIEW row's `risk_decision` is either `NULL` (never scored) or `allow` -- **no REVIEW row ever
carries a deny verdict**, so there is no risk narrative to explain. Asking a model to write prose
about `order_budget_exhausted` produces a story about an internal quota counter in the register of a
risk finding, which is worse than saying nothing.

| Row | Treatment |
| --- | --- |
| `status = 'final'` and `risk_decision = 'deny'` | LLM explanation |
| `action = 'review'` | Deterministic template from the `fail_safe_reason` enum. No API call |
| `action = 'allow'` | Nothing |

`status = 'final'` is defence in depth rather than a fix: preflight assigns `risk_decision` and
`status = 'final'` inside one transaction, so a `pending` row cannot carry a deny verdict today.
The predicate asserts it anyway, because it is exactly where that assumption would silently break.

### The input contract has no injection surface

Every field sent is a number, a bool, or a member of a closed enum. **Not one free-text field is
sent**, so there is no substring of the payload a user can author. This is the anti-injection
design: not a filter that strips dangerous input, but an input space in which no expressible input
is dangerous.

Sent: the calibrated probability and locked threshold at full stored precision, whether the
probability exceeds the threshold, `risk_decision`, `action`, `fail_safe_reason`, `amount_subunits`,
`currency`, `channel`, and the ledger co-occurrence counts already shown in case detail.

Never sent: **any pseudonymous token**. `EntityAggregate` has no field capable of holding one, so a
`dev_...` value cannot reach the model even by mistake. Event, checkout, merchant, session and
instrument identifiers, order ids and absolute timestamps are likewise absent; `span_seconds`
carries duration without pinning a moment. The prompt asks the model to write "this device".

### Output is structured, cross-checked, then sanitised

The model returns JSON against an enum-constrained schema, validated by Pydantic before anything is
stored. `factors` is a **closed enum, not prose** -- the model selects codes and RiskLoom renders
the human sentence from its own template and its own numbers, so an invented contributing factor is
not filtered out, it is unrepresentable.

Five stages, and failing any one stores `rejected` and displays nothing:

1. **Schema.** Unknown key, missing field, wrong type or out-of-enum code fails here.
2. **Factor support.** Every selected code must be entailed by the input. A model reporting prior
   denials on a device whose `denied_count` is zero is rejected with `unsupported_factor`.
3. **Numeral cross-check.** See below.
4. **Forbidden content.** Pseudonymous token patterns, PII shapes, markup, `javascript:`.
5. **Bounds.** Length caps on summary, caveat, factor count and total payload.

### The numeral rule: lossless renderings only

The rule is that **an exact, lossless re-rendering of a supplied value is permitted; a lossy
approximation of it is not.** One rule governs every field rather than one rule per field.

| Supplied | Accepted | Rejected |
| --- | --- | --- |
| `amount_subunits = 25000` | `25000`, `25,000`, `250.00`, `250` | `260.00` |
| `span_seconds = 201` | `201`, and the parts `3` and `21` | `3.35`, `200` |
| probability `0.007053679692244301` | that string verbatim; its exact percentage `0.7053679692244301` | `0.0071`, `0.71%`, `0.0070`, `0.007` |

A *truthful but rounded* restatement is rejected exactly as an invented figure is, and truncation is
rejected for the same reason rounding is. This is deliberate. The locked threshold carries nineteen
significant decimals and Gate C1 established that a real tie-cluster of scored rows sits one unit in
the last place below it. A panel permitting "roughly 0.71%" beside a threshold of
`0.0033862949155182734` would reintroduce precisely the display rounding the project bans everywhere
else. The prompt therefore asks for verbatim quotation or, preferably, qualitative wording -- both
exact values are already rendered in full precision immediately above the panel.

Cardinal number words from two upward are checked the same way. "one" and "a" are not, because they
are overwhelmingly idiomatic and checking them would reject far more true statements than false
ones. That limit is stated rather than hidden.

### Fail-safe and isolation

`failed` and `rejected` are distinct stored states: one is the model not answering, the other the
model answering with something the checks refused to trust. Neither affects any decision, order or
ledger row, and an integration test snapshots `risk_decisions` and `review_items` across a success,
a failure and a rejection and asserts byte-equality. **There is no retry loop** -- a retry is a
fresh, human-initiated request consuming one attempt and one unit of budget.

Isolation is enforced in both directions, statically over the AST and transitively in a fresh
interpreter. The generation package imports no ORM and holds no session, so it *cannot* write to
`risk_decisions`: the capability is absent rather than withheld.

### Bounded spend

```powershell
uv run alembic upgrade head
uv run uvicorn riskloom.main:app --host 127.0.0.1 --port 8000
```

Three independent caps: `RISKLOOM_GEMINI_MAX_CALLS_PER_PROCESS` (default 5, counting *attempts*
like the Razorpay order budget), three attempts per decision, and a uniqueness claim taken before
any outbound call so a concurrent duplicate is refused rather than spending a second unit. Every
call in the automated suite is faked; a guard test proves no test can construct an un-mocked client
and that test settings carry no key.

The client is a thin `httpx` adapter rather than the official `google-genai` SDK. The SDK is the
current official package, but it brings six new runtime dependencies for a single POST, retries
internally through tenacity, auto-discovers an ambient `GEMINI_API_KEY`, and raises exceptions that
can carry upstream response bodies -- three behaviours that conflict with standing invariants. The
wire format is confined to one module behind a `Protocol`, so swapping in the SDK later is a
contained change.

The model id is a single setting, `RISKLOOM_GEMINI_MODEL`, defaulting to `gemini-3.6-flash`. That
default was not chosen from documentation: a probe call against the live API rejected
`gemini-2.5-flash` with "no longer available to new users. Please update your code to use
models/gemini-3.6-flash". Older ids such as `gemini-2.0-flash` are shut down entirely. Verify the
current id before changing this value rather than trusting a published list.

### Known limitations

- **`POST` is a narrow exception to Day 7's read-only dashboard.** It writes exactly one row to
  `risk_decision_explanations` and touches nothing else; it cannot alter a decision, create an order
  or move a review item. Review-item mutation remains deferred.
- **The explanation is enrichment, not evidence.** It is generated after the fact from stored
  aggregates and played no part in the decision. The panel says so permanently.
- **No API key means no generation.** Without `RISKLOOM_GEMINI_API_KEY` the endpoint answers 503 and
  the panel reports the feature unconfigured. Startup is unaffected.
- **Free-tier terms may permit training on submitted content.** This is exactly why the input
  contract is aggregates and enums only: no token, no identifier and no timestamp leaves the process.

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
