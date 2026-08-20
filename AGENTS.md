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
