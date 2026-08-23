# RiskLoom engineering instructions

These instructions apply to the entire repository.

## Platform and architecture

- Use Python 3.11 and uv. Keep the distribution and import package named `riskloom`.
- Preserve the FastAPI modular monolith. Do not introduce services or infrastructure that are not
  required by the current task.
- Use SQLAlchemy 2 async APIs with PostgreSQL. Never use SQLite as a database-behaviour substitute.
- Make every schema change through an Alembic migration. Never create tables during app startup.
- Keep external integrations behind typed adapters. The Razorpay Orders client is internal only.

## Payment safety and privacy

- RiskLoom is defense-only and shadow-mode. It must not autonomously block, capture, refund, or
  otherwise decide payments.
- Support only Razorpay test-mode API keys. Reject key IDs that do not start with `rzp_test_`.
- Never commit, print, log, or return API key secrets, webhook secrets, authorization headers,
  signatures, webhook bodies, or upstream response bodies.
- Never store raw webhook bytes in PostgreSQL, logs, temporary files, exceptions, or reports.
- Verify webhook HMAC-SHA256 signatures over the exact raw request bytes before parsing JSON.
- Keep raw bytes only in request-scoped memory long enough to verify and hash them.
- Persist only an explicitly allowlisted audit projection and the SHA-256 digest. Unknown and
  free-form fields are untrusted and must be excluded.
- Never store raw card numbers, CVV, email addresses, phone numbers, unredacted network addresses,
  cardholder names, VPA values, free-form notes, descriptions, token data, or acquirer data.
- Use only synthetic reserved data in fixtures and examples. Never use production credentials or
  real payment information.

## Webhook invariants

- Use `X-Razorpay-Event-Id` as the provider idempotency key and enforce uniqueness in PostgreSQL.
- Insert the audit event and normalized business observation in one transaction.
- Keep payment observations append-only. Do not add a mutable current-status projection without an
  explicit transition and reconciliation design.
- A duplicate event must never create a duplicate business effect.
- Valid unsupported events are audited as ignored. Retryable processing failures must roll back and
  return non-2xx.

## Live scoring invariants

- Live scoring calls the unmodified Day 3 `FeatureEngine.process()`. There must never be a second
  implementation of any feature. The train/serve parity test drives the real
  `extract_feature_dataset` on one side and the online host on the other and requires all 75
  features to match exactly.
- Risk is decided by the locked Day 4 model and its single `decision_threshold` only. Compare the
  full float64 probability against the full float64 threshold from `model.json`. The
  `Numeric(20, 18)` ledger column is storage and audit only: it rounds the locked threshold to
  `0.003386294915518273`, which is exactly a real tie-cluster probability, so a decision made from
  the stored value would flip every row in that cluster.
- Neither Gate C1 policy band may be reachable from a live decision. `riskloom.serving`,
  `riskloom.api` and `riskloom.services.preflight` must not import `riskloom.policy` or
  `riskloom.modeling.training`, which reaches a band through its boundary diagnostic.
- REVIEW is an operational fail-safe tier, never a risk band. It is reached only when a decision
  cannot be safely completed: feature computation failed, scoring failed, order creation failed, or
  the process order budget is exhausted. No second risk threshold may exist.
- Serialise the read-then-update of rolling state with one process-wide lock. Per-merchant locks
  are unsound because device, network and instrument indices are shared across merchants. Hold no
  I/O inside the lock.
- Assign event timestamps server-side, strictly increasing, under the lock. Never accept a
  client-supplied timestamp: the engine requires strict monotonicity, and a caller must not be able
  to replay stale timestamps into rolling state.
- Live feature state is in-memory only and does not survive a process restart. Do not add a
  persistence layer without an explicit gate.
- Claim the idempotency key before touching the engine, so a retry cannot advance state twice or
  create a second order. A crashed run leaves a `pending` row and the retry is refused with 409;
  there is no auto-recovery.
- Cap Razorpay order-creation attempts per process. The cap counts attempts, not successes:
  the budget is taken before the upstream call, so a rejected attempt still consumes one and
  the counter can exceed the number of orders that exist. Past the cap an ALLOW fail-safes to
  REVIEW without calling Razorpay at all.
- Only `rzp_test_` keys, and nothing in this path may weaken Day 1 webhook signature verification,
  idempotency or redaction.

### Known limitation: live-serving accuracy is not measured to held-out standard

At preflight an attempt's outcome does not exist yet, so live serving advances state with every
attempt recorded as authorized. The 57 outcome-independent features are exact; the 18
failure-derived ones read low. This is quantified, not assumed: replaying the 9,000-event
policy-validation batch through the locked model under both assumptions gives recall 0.856 -> 0.600,
precision 0.577 -> 0.184, false-positive rate 1.28% -> 5.43%, and total cost 763 -> 2,279 units
(+199%). The degradation is material. Gate B2's held-out numbers describe offline scoring with true
outcomes and must not be quoted as live-serving accuracy. Webhook-driven failure reconciliation is
named future work; reproduce the measurement with `riskloom.analysis.blindspot`.

## Decision-boundary invariants

- Probability parity does not imply decision parity. The modeling parity gate compares calibrated
  probabilities within `parity_absolute_tolerance`; a difference far below that tolerance can still
  flip the discrete allow/deny decision for every row whose probability ties exactly at the
  threshold. Never treat a passing parity check as evidence that two implementations agree on
  decisions.
- A gradient-boosting model with shallow trees can legitimately produce very few distinct
  calibrated probabilities on a skewed feature set. The locked Day 4 model produces 15 distinct
  values across 7,952 `policy_selection` rows, so probabilities arrive in large ties and a
  threshold landing on a tie-cluster boundary is a recurring structural risk, not a one-off fluke.
- The locked Day 4 threshold sits one unit in the last place above a tie cluster of eight
  `policy_selection` rows (one attack, seven legitimate). Portable JSON inference therefore allows
  those eight rows, while `training_report.json` recorded them as denied. This is an internal
  consistency gap between that report and portable-model behaviour on `policy_selection` only. No
  published held-out evaluation number is affected: Gate B2 computed every figure through portable
  inference, so those numbers correctly describe deployed behaviour.
- The Day 4 artifact is not retroactively re-locked for this. The condition is instead monitored:
  `validate-model` reports a non-fatal `decision_boundary` diagnostic naming whether the threshold
  is an observed probability, the nearest observed value below it, and how many attack and
  legitimate rows tie there. It is a diagnostic and must never become a pass or fail gate against
  the already-accepted model.
- When a future re-lock or retraining run selects a threshold, prefer a value strictly between two
  adjacent observed probabilities over one landing exactly on an observed value, and check the
  diagnostic before accepting the result.

## Cost-aware policy invariants

- Keep `riskloom.policy` isolated from ground-truth labels. It must not import the simulation label
  module directly or transitively, and no policy source may name `scenario_type`, `campaign_id`,
  `is_attack`, `split`, or `generator_metadata`. Routing inputs are the locked model's calibrated
  probability and, where needed, the same 75 causal features available at inference time.
- Keep label-bearing orchestration in `riskloom.modeling.policy_ops`. Labels may score a decision
  that has already been made; they may never be an input to making one.
- Fit both band thresholds on `policy_selection` only. That partition's Day 4 role was already
  decision-rule selection, so choosing two thresholds is an extension of that role, not new
  leakage. Never fit against held-out or counterfactual-validation rows.
- Total cost is `FN * false_negative_cost_units + FP * false_positive_cost_units +
  N_review * review_cost_units`. Inherit 25 and 1 unchanged from the locked Day 4 configuration so
  both policies are scored identically. `review_cost_units` is an abstract operational weight
  charged for every reviewed event regardless of its true label; it is a policy choice, never a
  currency claim. A reviewed event is neither a false positive nor a false negative.
- Sweep both thresholds deterministically over observed probabilities, always including the
  incumbent threshold as a candidate. Apply the extended tie-break ladder: cost, false positive,
  true positive, review count, upper threshold, lower threshold, then stable candidate index.
  Precision is deliberately absent because it is determined by true positive and false positive.
- Validate counterfactually on a `policy-validation` profile batch generated with a fresh seed and
  a chronologically disjoint window. Refuse any batch whose dataset ids, artifact hashes, or
  configuration fingerprint match the locked development contract.
- Never auto-activate a policy. Approval requires an explicit human flag and is refused whenever
  the banded policy fails to beat the incumbent cost, exceeds the configured false-positive-rate
  ceiling, or the validation batch is below the configured minimum rows and attacks. Publish the
  comparison honestly whether the policy wins or loses.
- Generated policy artifacts belong under ignored `artifacts/policy/` and must not be committed.

## Quality gates

- Add or update unit and PostgreSQL integration tests for every behavior change.
- Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src`, and the complete
  pytest suite before handoff.
- Verify Alembic upgrade, downgrade, re-upgrade, and `alembic check` for schema work.
- Do not create a commit unless the user explicitly asks for one.

## Deterministic simulation invariants

- Keep `riskloom.simulation` isolated from the API, database, Razorpay integration, and production
  webhook path. Simulation changes must not add endpoints, migrations, persistence, or network I/O.
- Synthetic events and evaluation labels are separate strict schemas. Replay consumes only the
  event schema and must never import, accept, or inspect labels or expose an HTTP, socket, URL,
  Razorpay, or other remote transport surface.
- Only `event_id` is globally unique. Entity tokens are intentionally reusable; do not add global
  uniqueness assertions for merchants, checkouts, customers, devices, networks, sessions, or
  payment instruments.
- Preserve the event invariant in both directions: authorized events have
  `failure_category=null`, and failed events have a non-null allowed failure category.
- Build all IDs with deterministic UUIDv5 and serialize the UUID through `uuid.hex`. Derive
  independent PRNG streams with SHA-256 from the seed, split, and component name.
- Preserve exact integer scenario quotas in every split. Retry chains and campaigns must consume
  their event allocations exactly even though they group events.
- Preserve simulation algorithm and configuration `1.0.0` byte-for-byte: retain its UUID inputs,
  PRNG stream material and draw order, campaign path, effective-configuration representation,
  artifacts, and historical validation. It never accepts a configuration fingerprint or campaign
  placement. Schema `1.1.0` binds every identifier and PRNG stream to the canonical SHA-256 of the
  complete effective configuration and records that fingerprint in provenance.
- Keep the generic `1.1.0` placement schema reusable for non-development tests, but lock the tracked
  development contract before generation. Development calibration has ten equal 34-event
  campaigns. Compute its protected
  timestamp as `start + floor(duration_ms * 6000 / 10000)`: before means strictly earlier, and
  after means equal or later. Place at least five complete campaigns on each side and never allow a
  campaign to cross the boundary.
- Select schema `1.1.0` profile contracts only from the canonical effective configuration, never a
  filename or output path. `development` is the exact locked contract. `smoke` permits at most 3,000
  total events, 1,400 events and 28 attacks per split, 60 attacks total, six total days, four days
  per split, and ten campaigns per split; protected placement permits at most five campaigns per
  side and 4,096 attempts per campaign. Apply the shared contract during model validation, config
  loading, generation preflight, and artifact validation. Do not apply these new bounds to `1.0.0`.
- At every public generation entry, rebuild a strict `GeneratorConfig` from
  `config.model_dump(mode="python")` before any output-path or staging operation. Apply the shared
  profile contract to that snapshot and use only the snapshot afterward. Reject mutated Boolean,
  fractional, negative, malformed, unknown, downgraded, or structurally incompatible values.
- Campaign placement must use independent seeded irregular sampling across the permitted time
  intervals, never equal periodic slots. Constrained campaign windows cannot overlap and must keep
  the configured five-minute gap.
- Preserve the controlled test-shift policy from configuration and validate it using integer
  cross-products: test attack unique-device/event and unique-session/event ratios are at least 2x
  both train and calibration, so events/device and events/session decrease. Keep attack network
  uniqueness at or below 5,000 basis points and network presence at or above 9,000 basis points.
  Do not substitute or claim an instruments-per-device direction for this invariant.
- Canonical artifacts use UTF-8, LF endings, compact JSON, sorted keys, and no uncontrolled float,
  wall-clock timestamp, host data, locale, absolute path, or output-directory-dependent identity.
  Sort every distribution derived from mappings. The manifest hashes events, labels, and report,
  never itself.
- Treat staged, validated, manifest-last publication as a completeness marker, not a fully atomic
  multi-file transaction. Refuse unsafe locations and unknown-file overwrites without adding a
  generalized filesystem-management layer.
- Recursively reject the explicit prohibited field set: card number/PAN, CVV/CVC, expiry, email,
  phone/contact, address/billing address, VPA/UPI ID, IP address, and raw payload. Do not reject the
  typed `payment_instrument_token`. Do not claim to detect arbitrary human names; keep all name
  fields, configuration, generators, and dependencies absent.
- Generated datasets belong under the ignored `artifacts/simulations/` tree and must not be
  committed. Use tiny deterministic configurations in pytest. Generate the 100,000-event profile
  only as an explicit manual verification command.

## Causal feature-engine invariants

- Keep `riskloom.features` isolated from labels, simulation generation and validation modules,
  FastAPI, databases, Razorpay, network transports, subprocesses, dynamic imports, and model or
  policy code. Feature extraction accepts only the model-visible `CheckoutAttemptEvent` stream.
- Preserve the version 1.0.0 schema of exactly 75 integer features. Event ID and occurrence time
  are join/audit metadata, not model features. A name, formula, sentinel, window, boundary, or
  ordering change requires an explicit feature schema and/or engine version increment.
- Process every event in this order: validate typed input and strict `(occurred_at, event_id)`
  ordering; evict expired state; compute and validate its feature record from prior state; update
  state with the current event; commit the ordering key; return the precomputed record. A failed
  feature construction must not update state.
- Use event-time windows with `(current_time - window, current_time]` semantics over prior processed
  events. An event exactly on the left boundary is expired. Same-timestamp events are causal in
  event-ID order. Do not reset at dataset split boundaries.
- Keep causal state bounded to 3,600 seconds with globally evicted deques and reference-counted
  relationship counters. Do not retain global event IDs or expired entity buckets. Missing device
  or network tokens never form a shared null bucket and always produce zero history features.
- Read current outcome only while updating state for future events. Never use current outcome,
  failure category, currency, entity tokens, labels, scenarios, campaigns, splits, generator
  metadata, risk scores, thresholds, decisions, or predictions as model features.
- Canonical feature artifacts use UTF-8, LF, compact sorted-key JSON and integer-only statistics.
  Validate source events and feature rows in streaming lockstep. Hash exact source bytes and reject
  changes between hashing, extraction, validation, and publication.
- Publish only `features.jsonl`, `report.json`, and `manifest.json` through a scoped sibling staging
  directory. Overwrite only a fully valid RiskLoom feature dataset for the same exact source bytes
  and effective configuration. Replace the manifest last and never remove unknown content.
- Generated feature artifacts belong under ignored `artifacts/features/` and must not be committed.
  The 100,000-event development extraction is a manual verification, not a pytest fixture.

## Offline modeling invariants

- Keep `riskloom.modeling` offline and isolated from FastAPI, databases, Razorpay, HTTP, sockets,
  subprocesses, dynamic imports, and remote transports. It may use only the standard library,
  Pydantic, NumPy, scikit-learn, the strict feature schema, and the minimum simulation schemas
  needed to validate labels and the locked development contract.
- Train only against the exact approved development simulation and feature identities and hashes.
  Validate labels through the Day 2 manifest and exact label-file hash; training does not require or
  open raw events. Refuse any profile other than the exact `development` schema 1.1.0 contract.
- Subdivide calibration at `start + floor(duration_ms * 6000 / 10000)`. `calibration_fit` is
  strictly before the timestamp; `policy_selection` is equal or later. Fit preprocessing, class
  weights, and candidates on train only; fit Platt calibration on calibration-fit only; select the
  candidate and cost-sensitive threshold on policy-selection only.
- Training must discard held-out features and must not validate, read, store, or report held-out
  targets. `validate-model` may sample held-out feature vectors for inference parity only after a
  locked model has passed canonical schema, identity, and hash validation. Official evaluation is
  a separate fit-free command and requires explicit review approval.
- Persist models only as strict canonical data-only JSON. Never use pickle, cloudpickle, joblib,
  arbitrary object deserialization, callbacks, code loading, or estimator persistence. Validate
  tree topology, feature order, class order, finite values, and dimensions before inference.
- Compare the in-memory winning estimator with portable inference on deterministic train,
  calibration-fit, and policy-selection samples before publication. `validate-model` independently
  retrains and repeats parity checks. Publish model, aggregate-only report, and manifest through a
  process-owned sibling staging directory, with the manifest last and no overwrite surface.
- Keep average precision distinct from trapezoidal PR-AUC. Use `null` for rates over empty
  hard-negative slices. Include completely missed campaigns as zero flagged events in campaign
  distributions. Never place raw event/entity/campaign IDs or per-event predictions in reports.
- Generated model and evaluation artifacts belong under ignored `artifacts/models/` and
  `artifacts/evaluations/`. Do not run the official development held-out evaluation unless the user
  explicitly authorizes that separate gate.

## Dashboard invariants

- The coordination graph layout lives in `riskloom.serving.coordination` as pure, pytest-covered
  Python: connected-component hub ordering, ellipse placement scaled to the requested canvas,
  centroid-based event placement, and deterministic relaxation. It is not client-side JavaScript.
  Any change to graph layout belongs there, and its tests assert on-screen geometry, so a
  regression in them is a visible regression. Do not reintroduce positioning logic in `static/`.
- Layout must stay deterministic: no random source, ordering derived from sorted node identifiers,
  and any jitter hashed from the identifier. The same graph and canvas always produce the same
  coordinates.
- Build graph topology first and lay it out second. A single pass that positions each event as its
  hub is emitted anchors a shared event to whichever hub sorts first, which is what previously
  collapsed the graph into a corner of the canvas.
- `canvas_width` and `canvas_height` are request inputs that affect coordinates only. They must
  never change which nodes or edges exist, and they are not persisted.
- Keep the dashboard read-only and GET-only. It must not gain a mutation path for review items,
  decisions, or orders without an explicit gate: doing so would pull it inside the safety boundary
  Days 4-6 were held to.
- Display only stored pseudonymous tokens and aggregates from the explicit response allowlist.
  Never display or derive a feature name, feature value, policy band, or recomputed score. Live
  ledger co-occurrence and offline `evaluation.json` campaign metrics must stay visibly distinct;
  the offline figures come from ground-truth labels that do not exist for live traffic.
- Treat a missing `artifacts/evaluations/` artifact as an ordinary state. The model endpoint answers
  404 and the client renders an unavailable state; startup must never depend on it.

## Explanation invariants

- The LLM explains, it never decides. `riskloom.explanations` must never influence
  `risk_decision`, `action`, `calibrated_probability`, `decision_threshold`, or any other value in
  `risk_decisions`, and must never gate, delay, or alter a live decision.
- Keep `riskloom.explanations` pure: no ORM, no session, no FastAPI, and nothing from
  `riskloom.serving`, `riskloom.services`, `riskloom.policy`, `riskloom.modeling` or
  `riskloom.features`. Write isolation is by construction -- the package cannot reach a database,
  so it cannot write to the ledger. All persistence lives in `riskloom.services.explanations`.
- Enforce isolation in both directions and at both levels: an AST check over the sources, and a
  fresh-interpreter `sys.modules` probe. The decision path must never load the explanation package,
  and the explanation package must never load the decision path or `riskloom.db`. Do not place
  explanation dependencies in the shared `api/dependencies.py`: that made the package reachable from
  `api/routes/checkout.py` and the isolation test correctly rejected it.
- Generation is lazy and human-initiated. Never generate from the preflight path, never retry
  automatically, and never add a background queue. A retry is a fresh request that consumes one
  attempt and one unit of budget.
- Only a finalised DENY is eligible. REVIEW is an operational fail-safe whose `risk_decision` is
  null or `allow`, so it carries no risk narrative; render its `fail_safe_reason` deterministically
  instead. Keep `status == 'final'` in the eligibility predicate even though preflight cannot
  currently produce a pending deny.
- Send only numbers, bools and closed enums. Never send a free-text field, a pseudonymous token, an
  identifier, or an absolute timestamp. The absence of a free-text field is the anti-injection
  property, so do not add one.
- Keep `factors` a closed enum that the model selects from and RiskLoom renders. Never store or
  display model-authored factor text. Every selected code must be entailed by the input.
- Numerals in model prose are checked against exact, lossless re-renderings of supplied values only.
  Rounded and truncated restatements are rejected deliberately. If manual verification shows
  repeated `unsupported_number` rejections, tighten the prompt -- never widen the tolerance.
- Never log, store, or return an upstream response body, header, or API key. Failure reasons are
  short stable identities. `failed` and `rejected` are distinct states and must stay distinct.
- Bound real spend with all three caps: process call budget, per-decision attempts, and the
  uniqueness claim taken before the outbound call. Automated tests must never make a real call.
- The dashboard stays read-only apart from this single `POST`, which writes only
  `risk_decision_explanations`. Review-item mutation remains out of scope.

## Resilience, drift, and packaging invariants

- Failure-injection code proves existing behaviour; it must never implement a fail-safe. If a
  scenario needs new production logic to pass, the gap is in production, not in the drill.
- Keep `scripts/failure_drill.py` a client of the service. It must not reach around the API to
  manipulate the database, and storage scenarios must ask the operator to interrupt the database
  explicitly rather than doing it invisibly.
- Preflight must fail closed on any storage failure: 503 `storage_unavailable`, never a 200 ALLOW
  and never a partial ledger write. Catch `OSError` as well as `SQLAlchemyError` -- a frozen
  database surfaces a bare `TimeoutError` that SQLAlchemy does not wrap, and without it the request
  answers 500. A frozen database is not bounded server-side; do not claim otherwise.
- If storage fails after an order was created, log `preflight_ledger_write_failed` with the order
  id. The row stays `pending` and there is no auto-recovery.
- `riskloom.drift` is informational only and must never influence a decision, a threshold, or a
  model parameter. It holds no session and imports no ORM, so it cannot write to any table; all
  querying lives in `riskloom.services.drift` and is read-only. Enforce isolation in both
  directions with an AST check and a fresh-interpreter probe, and keep drift dependencies off the
  shared `api/dependencies.py`.
- The PSI reference comes only from the published per-bin counts in `evaluation.json`. Never open,
  re-score, or re-derive the protected test partition.
- `PSI_EPSILON` and `PSI_MINIMUM_ROWS` are load-bearing constants. The epsilon moves the result
  across a band boundary, so changing it requires re-recording the sensitivity table asserted in
  the tests. Never report a band below the minimum row count.
- Always report per-bin contributions alongside the scalar. The locked reference is degenerate --
  97.78% in one bin, five empty bins, the threshold inside the first bin -- so a lone number is
  routinely dominated by one bin and reads as drift when it is not.
- Never verify a PSI figure by trusting a previously written one. Recompute it from the formula and
  cross-check against a second implementation; an earlier draft of this gate recorded exactly twice
  the correct value by doubling the per-bin difference.
- No secret may reach an image layer. `.dockerignore` excludes `.env`; secrets arrive through the
  environment at runtime. A blank key counts as absent.
- Migrations run in a one-shot `migrate` service that exits before the app starts. The application
  never creates tables.
- Locked artifacts are runtime inputs mounted read-only, never baked into the image and never
  committed. Do not add a flag that skips startup model binding to make a fresh clone start.
