"""Explanation endpoints.

Day 7's dashboard router is GET-only and stays that way. This is a separate router carrying one
narrow, justified exception: ``POST`` here writes exactly one row to ``risk_decision_explanations``
and touches nothing else. It cannot alter ``risk_decision``, ``action``, ``calibrated_probability``,
``decision_threshold``, ``status`` or ``razorpay_order_id``; it cannot create an order, and it
cannot move a review item. It is additive enrichment about a decision that is already final, which
is categorically different from the deferred review-item mutation capability.

Three independent caps bound real spend: a process-wide call budget, a per-decision attempt cap,
and a uniqueness claim that refuses a concurrent duplicate before any outbound call.
"""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.api.dependencies import get_app_settings, get_session
from riskloom.core.config import Settings
from riskloom.dashboard.schemas import ExplanationView
from riskloom.explanations.client import GeminiClientProtocol
from riskloom.services.explanations import (
    ExplanationBudget,
    ExplanationRefused,
    generate,
    latest,
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


# These providers live here rather than in the shared dependency module on purpose. Placing them
# there made ``riskloom.explanations`` reachable from ``api/routes/checkout.py`` -- the live
# decision path -- through a single shared import, which the isolation test correctly rejected.
def get_explanation_budget(request: Request) -> ExplanationBudget:
    return cast(ExplanationBudget, request.app.state.explanation_budget)


def get_explanation_client(request: Request) -> GeminiClientProtocol | None:
    """``None`` when no API key is configured, which the route reports as 503."""

    return cast(GeminiClientProtocol | None, request.app.state.explanation_client)


_STATUS_FOR = {
    "not_found": 404,
    "not_eligible": 422,
    "in_progress": 409,
    "attempts_exhausted": 409,
    "budget_exhausted": 429,
    "not_configured": 503,
}


def _error(request: Request, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=_STATUS_FOR.get(code, 400),
        content={"error": {"code": code, "request_id": request.state.request_id}},
    )


@router.get("/decisions/{decision_id}/explanation", response_model=None)
async def read_explanation(
    request: Request,
    decision_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ExplanationView | JSONResponse:
    view = await latest(session, decision_id)
    if view is None:
        return _error(request, "not_found")
    return view


@router.post("/decisions/{decision_id}/explanation", response_model=None, status_code=201)
async def create_explanation(
    request: Request,
    decision_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    budget: Annotated[ExplanationBudget, Depends(get_explanation_budget)],
    client: Annotated[GeminiClientProtocol | None, Depends(get_explanation_client)],
) -> ExplanationView | JSONResponse:
    """Generate an explanation.

    The request body is empty on purpose: there is nothing for a caller to supply, which is itself
    a property of the injection-surface design.
    """

    try:
        view = await generate(session, client, budget, decision_id, settings.gemini_model)
    except ExplanationRefused as refusal:
        return _error(request, refusal.code)

    # A stored failure or rejection is a successful request whose generation did not succeed.
    if view.status in {"failed", "rejected"}:
        return JSONResponse(status_code=202, content=view.model_dump(mode="json"))
    return view
