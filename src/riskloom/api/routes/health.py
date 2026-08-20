import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from riskloom.api.dependencies import get_app_settings, get_database
from riskloom.core.config import Settings
from riskloom.db.session import Database

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.get("/health/ready")
async def readiness(
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> JSONResponse:
    try:
        async with asyncio.timeout(settings.database_connect_timeout_seconds):
            await database.ping()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": {"database": "unavailable"}},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "checks": {"database": "ok"}},
    )
