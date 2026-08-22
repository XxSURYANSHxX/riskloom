import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from riskloom.api.router import api_router
from riskloom.core.config import Settings, get_settings
from riskloom.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    request_id_from_header,
)
from riskloom.db.session import Database
from riskloom.integrations.razorpay.client import RazorpayOrdersClient
from riskloom.services.preflight import OrderBudget
from riskloom.serving.engine_host import OnlineFeatureEngine
from riskloom.serving.model_host import load_serving_bundle


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or get_settings()
        configure_logging(resolved_settings.log_level)
        app.state.settings = resolved_settings
        app.state.database = Database(resolved_settings)
        # Fail closed: the service must not serve a decision unless it can prove it is bound to
        # the locked feature configuration and the locked model those features were trained with.
        app.state.serving_bundle = load_serving_bundle(
            feature_config_path=resolved_settings.feature_config_path,
            modeling_config_path=resolved_settings.modeling_config_path,
            model_directory=resolved_settings.risk_model_directory,
            feature_manifest_path=resolved_settings.feature_manifest_path,
        )
        # One warm engine and one HTTP client for the whole process lifetime.
        app.state.engine_host = OnlineFeatureEngine(app.state.serving_bundle.feature_config)
        app.state.orders_client = RazorpayOrdersClient(resolved_settings)
        app.state.order_budget = OrderBudget(
            limit=resolved_settings.razorpay_max_orders_per_process
        )
        try:
            yield
        finally:
            await app.state.orders_client.close()
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

    # The dashboard is a static, read-only client. StaticFiles serves GET and HEAD only, so
    # mounting it adds no mutation surface. Missing directory is tolerated so the API still starts
    # in environments where the client was not shipped.
    static_directory = (settings or get_settings()).dashboard_static_directory
    if static_directory.is_dir():
        application.mount(
            "/dashboard",
            StaticFiles(directory=static_directory, html=True),
            name="dashboard",
        )

    return application


app = create_app()


def run() -> None:
    uvicorn.run("riskloom.main:app", host="127.0.0.1", port=8000)
