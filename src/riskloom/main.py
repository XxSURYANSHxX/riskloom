import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response

from riskloom.api.router import api_router
from riskloom.core.config import Settings, get_settings
from riskloom.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    request_id_from_header,
)
from riskloom.db.session import Database


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or get_settings()
        configure_logging(resolved_settings.log_level)
        app.state.settings = resolved_settings
        app.state.database = Database(resolved_settings)
        try:
            yield
        finally:
            await app.state.database.close()

    application = FastAPI(
        title="RiskLoom",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @application.middleware("http")
    async def structured_request_logging(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request_id_from_header(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        bind_request_context(request_id)
        started = time.perf_counter()
        logger = structlog.get_logger("riskloom.http")
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1_000, 2),
                error_type=type(exc).__name__,
            )
            raise
        else:
            response.headers["X-Request-Id"] = request_id
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1_000, 2),
            )
            return response
        finally:
            clear_request_context()

    application.include_router(api_router)
    return application


app = create_app()


def run() -> None:
    uvicorn.run("riskloom.main:app", host="127.0.0.1", port=8000)
