"""Live checkout-preflight scoring.

This route is an intentional, approved expansion of the public API surface, which every prior gate
held fixed at two health endpoints plus the Razorpay webhook. It is scope change by decision, not
scope creep.
"""

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.api.dependencies import (
    get_app_settings,
    get_engine_host,
    get_order_budget,
    get_orders_client,
    get_serving_bundle,
    get_session,
)
from riskloom.core.config import Settings
from riskloom.integrations.razorpay.client import RazorpayOrdersClient
from riskloom.services.preflight import OrderBudget, PreflightPendingError, evaluate_preflight
from riskloom.serving.engine_host import OnlineFeatureEngine
from riskloom.serving.model_host import ServingBundle
from riskloom.serving.schemas import CheckoutPreflightRequest

router = APIRouter(prefix="/api/v1/checkout", tags=["checkout"])
logger = structlog.get_logger("riskloom.checkout")


def _error(request: Request, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "request_id": request.state.request_id}},
    )


@router.post("/preflight")
async def checkout_preflight(
    request: Request,
    payload: CheckoutPreflightRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    engine: Annotated[OnlineFeatureEngine, Depends(get_engine_host)],
    bundle: Annotated[ServingBundle, Depends(get_serving_bundle)],
    orders: Annotated[RazorpayOrdersClient, Depends(get_orders_client)],
    budget: Annotated[OrderBudget, Depends(get_order_budget)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> JSONResponse:
    try:
        # A wall-clock bound around the whole scoring path, for stalls that are cancellable.
        #
        # Measured behaviour, because the two outage shapes differ and it would be wrong to imply
        # otherwise. A database that *refuses* -- stopped, crashed, unreachable -- fails here in
        # about 3s and answers 503. A database that is *frozen* -- SIGSTOP, a hung host -- keeps
        # its socket open and answers nothing, and this timeout does not fire: SQLAlchemy's
        # greenlet bridge does not deliver the cancellation into asyncpg's blocked read, so the
        # request unwinds only once the connection dies. It still answers 503 and never a 200
        # ALLOW, so the decision stays fail-closed either way -- but the frozen case is not
        # bounded server-side, and that is a known limitation rather than a solved problem.
        async with asyncio.timeout(settings.database_command_timeout_seconds):
            decision = await evaluate_preflight(
                session,
                payload,
                engine=engine,
                bundle=bundle,
                orders=orders,
                budget=budget,
            )
    except PreflightPendingError:
        # A prior attempt claimed this event id and never finalised. There is no auto-recovery in
        # this gate; the caller is told plainly rather than being silently re-scored.
        return _error(request, 409, "decision_pending")
    except (SQLAlchemyError, OSError) as exc:
        # OSError as well as SQLAlchemyError, and this was found by injection rather than by
        # reading: pausing the database freezes the server process while the socket stays open,
        # so the failure surfaces as a bare TimeoutError (a subclass of OSError in 3.11) that
        # SQLAlchemy never wraps. Without this the request fell through to the generic handler and
        # answered 500 internal_error, which reads as "we broke" rather than "storage is gone".
        logger.error(
            "preflight_storage_failed",
            reason="database_error",
            error_type=type(exc).__name__,
        )
        return _error(request, 503, "storage_unavailable")
    except Exception:
        logger.error("preflight_failed", reason="internal_error")
        return _error(request, 500, "internal_error")

    return JSONResponse(status_code=200, content=decision.model_dump(mode="json"))
