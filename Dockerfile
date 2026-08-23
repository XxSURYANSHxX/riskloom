# RiskLoom application image.
#
# No secret is present at any layer. The Razorpay keys, the optional Gemini key and the database
# password all arrive at runtime through the environment, matching the existing Settings contract.
# `.dockerignore` excludes `.env` so it cannot be copied in even by accident.
#
# The locked model artifacts are deliberately NOT baked in. They are Git-ignored inputs, mounted
# read-only at runtime, so the image stays free of anything that would pin it to one trained model.

FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is cached until the lock changes.
# README.md is copied because pyproject declares it as the package readme and the build
# backend refuses to build the project without it.
COPY pyproject.toml uv.lock README.md ./
COPY src/riskloom/__init__.py src/riskloom/__init__.py
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

# Unprivileged by construction: the process cannot write to the application tree, and the mounted
# artifact directory is read-only from the compose side as well.
RUN useradd --create-home --uid 10001 riskloom

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder --chown=riskloom:riskloom /app/.venv /app/.venv
COPY --chown=riskloom:riskloom src ./src
COPY --chown=riskloom:riskloom alembic.ini ./
COPY --chown=riskloom:riskloom alembic ./alembic
COPY --chown=riskloom:riskloom configs ./configs
COPY --chown=riskloom:riskloom static ./static

USER riskloom

EXPOSE 8000

# Startup binds to the locked model and fails closed if it cannot. That is intentional: a container
# that exits with a `serving_*` identity is correct behaviour, not a packaging defect.
CMD ["uvicorn", "riskloom.main:app", "--host", "0.0.0.0", "--port", "8000"]
