from collections.abc import AsyncIterator
from typing import cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.core.config import Settings
from riskloom.db.session import Database


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    database = get_database(request)
    async for session in database.sessions():
        yield session
