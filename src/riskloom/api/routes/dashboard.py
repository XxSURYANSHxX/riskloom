"""Read-only dashboard API.

Every path on this router is registered with ``@router.get`` and nothing else. Starlette therefore
answers every other verb with 405 at the router, before any handler runs — enforcement by routing,
not by convention. A test asserts this for each endpoint.

Nothing here writes, and nothing here computes: each response is a projection of rows the live
scoring path already wrote, or of an artifact Gate B2 already produced.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.api.dependencies import get_app_settings, get_session
from riskloom.core.config import Settings
from riskloom.dashboard.artifacts import EvaluationUnavailableError, load_model_evaluation
from riskloom.dashboard.schemas import (
    CoordinationGraph,
    DecisionDetail,
    DecisionPage,
    LedgerSummary,
    ModelEvaluation,
)
from riskloom.services.dashboard import (
    DecisionFilter,
    build_coordination_graph,
    get_decision,
    list_decisions,
    summarise,
)
from riskloom.serving.coordination import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    MAXIMUM_CANVAS_HEIGHT,
    MAXIMUM_CANVAS_WIDTH,
    MINIMUM_CANVAS_HEIGHT,
    MINIMUM_CANVAS_WIDTH,
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

ActionFilter = Annotated[str | None, Query(pattern="^(allow|review|deny)$")]


def _not_found(request: Request, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": {"code": code, "request_id": request.state.request_id}},
    )


@router.get("/summary")
async def dashboard_summary(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> LedgerSummary:
    return await summarise(session)


@router.get("/decisions")
async def dashboard_decisions(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    action: ActionFilter = None,
    since: datetime | None = None,
) -> DecisionPage:
    return await list_decisions(
        session, DecisionFilter(limit=limit, offset=offset, action=action, since=since)
    )


@router.get("/decisions/{decision_id}", response_model=None)
async def dashboard_decision(
    request: Request,
    decision_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DecisionDetail | JSONResponse:
    detail = await get_decision(session, decision_id)
    if detail is None:
        return _not_found(request, "decision_not_found")
    return detail


@router.get("/coordination")
async def dashboard_coordination(
    session: Annotated[AsyncSession, Depends(get_session)],
    window_seconds: Annotated[int, Query(ge=60, le=86_400)] = 3_600,
    canvas_width: Annotated[
        int, Query(ge=MINIMUM_CANVAS_WIDTH, le=MAXIMUM_CANVAS_WIDTH)
    ] = DEFAULT_CANVAS_WIDTH,
    canvas_height: Annotated[
        int, Query(ge=MINIMUM_CANVAS_HEIGHT, le=MAXIMUM_CANVAS_HEIGHT)
    ] = DEFAULT_CANVAS_HEIGHT,
) -> CoordinationGraph:
    """Shared-token graph, laid out for a canvas of the given size.

    The dimensions default to a desktop panel so the endpoint is useful with no parameters at all;
    the dashboard measures its own panel and passes the real numbers, because the server cannot
    otherwise know how wide the viewport is. They affect coordinates only -- never which nodes or
    edges exist.
    """

    return await build_coordination_graph(session, window_seconds, canvas_width, canvas_height)


@router.get("/model", response_model=None)
async def dashboard_model(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ModelEvaluation | JSONResponse:
    """Offline held-out aggregates.

    The artifact is Git-ignored, so absence is an ordinary outcome: the endpoint answers 404 and
    the dashboard renders an explicit unavailable state rather than failing.
    """

    try:
        return load_model_evaluation(settings.evaluation_artifact_path)
    except EvaluationUnavailableError:
        return _not_found(request, "evaluation_unavailable")
