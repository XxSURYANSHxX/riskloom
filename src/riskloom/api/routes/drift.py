"""Drift endpoint.

GET-only, on its own router, following Day 7's pattern: every path here is registered with
``@router.get`` and nothing else, so Starlette answers 405 to every other verb by routing rather
than by convention.

Read-only in the strongest sense available: the handler issues a ``SELECT`` and returns arithmetic.
It writes nothing, and the module it calls cannot write anything.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.api.dependencies import get_app_settings, get_session
from riskloom.core.config import Settings
from riskloom.drift.schemas import DriftReport
from riskloom.services.drift import (
    DEFAULT_WINDOW_HOURS,
    MAXIMUM_WINDOW_HOURS,
    evaluate_drift,
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/drift")
async def dashboard_drift(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    window_hours: Annotated[int, Query(ge=1, le=MAXIMUM_WINDOW_HOURS)] = DEFAULT_WINDOW_HOURS,
) -> DriftReport:
    """Score-distribution stability against the locked held-out reference.

    Absence of the evaluation artifact is an ordinary state, not an error: the response carries
    ``status: reference_unavailable`` rather than a 404, because the endpoint still has something
    true to say about why there is no number.
    """

    return await evaluate_drift(session, settings.evaluation_artifact_path, window_hours)
