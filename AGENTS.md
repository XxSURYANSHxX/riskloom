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
