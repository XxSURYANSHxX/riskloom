from collections.abc import AsyncIterator
from typing import cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.core.config import Settings
from riskloom.db.session import Database
from riskloom.integrations.razorpay.client import RazorpayOrdersClient
from riskloom.services.preflight import OrderBudget
from riskloom.serving.engine_host import OnlineFeatureEngine
from riskloom.serving.model_host import ServingBundle


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    database = get_database(request)
    async for session in database.sessions():
        yield session


def get_engine_host(request: Request) -> OnlineFeatureEngine:
    return cast(OnlineFeatureEngine, request.app.state.engine_host)


def get_serving_bundle(request: Request) -> ServingBundle:
    return cast(ServingBundle, request.app.state.serving_bundle)


def get_orders_client(request: Request) -> RazorpayOrdersClient:
    return cast(RazorpayOrdersClient, request.app.state.orders_client)


def get_order_budget(request: Request) -> OrderBudget:
    return cast(OrderBudget, request.app.state.order_budget)
