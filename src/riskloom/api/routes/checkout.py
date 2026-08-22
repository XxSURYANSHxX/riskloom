"""Live checkout-preflight scoring.

This route is an intentional, approved expansion of the public API surface, which every prior gate
held fixed at two health endpoints plus the Razorpay webhook. It is scope change by decision, not
scope creep.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.api.dependencies import (
    get_engine_host,
    get_order_budget,
    get_orders_client,
    get_serving_bundle,
    get_session,
)
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
) -> JSONResponse:
    try:
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
    except SQLAlchemyError:
        logger.error("preflight_storage_failed", reason="database_error")
        return _error(request, 503, "storage_unavailable")
    except Exception:
        logger.error("preflight_failed", reason="internal_error")
        return _error(request, 500, "internal_error")

    return JSONResponse(status_code=200, content=decision.model_dump(mode="json"))
