# RiskLoom

RiskLoom is a defense-only, shadow-mode risk-management backend for detecting coordinated
card-testing activity. Day 1 establishes secure Razorpay test-mode webhook ingestion and an
internal Orders API adapter. It does not score, block, capture, or otherwise decide payments.

## Day 1 capabilities

- Python 3.11 FastAPI modular monolith.
- PostgreSQL persistence through SQLAlchemy 2 and Alembic.
- Exact-raw-body Razorpay webhook HMAC-SHA256 verification.
- Database-enforced idempotency using `X-Razorpay-Event-Id`.
- Append-only payment observations that tolerate out-of-order delivery.
- Allowlisted audit projections and exact-body SHA-256 digests; raw bodies are never persisted.
- Internal-only Razorpay test-mode Orders client using httpx.
- Structured logs, liveness/readiness checks, and PostgreSQL integration tests.

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
