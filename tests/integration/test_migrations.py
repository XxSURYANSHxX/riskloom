import asyncio

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command
from tests.conftest import make_alembic_config

pytestmark = pytest.mark.integration


async def _table_names(database_url: str) -> set[str]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return set(await connection.run_sync(lambda sync: inspect(sync).get_table_names()))
    finally:
        await engine.dispose()


def test_migration_upgrade_downgrade_reupgrade_and_drift(postgres_url: str) -> None:
    config = make_alembic_config(postgres_url)

    command.upgrade(config, "head")
    assert {"alembic_version", "webhook_events", "payment_observations"}.issubset(
        asyncio.run(_table_names(postgres_url))
    )

    command.downgrade(config, "base")
    assert asyncio.run(_table_names(postgres_url)) == {"alembic_version"}

    command.upgrade(config, "head")
    command.check(config)
