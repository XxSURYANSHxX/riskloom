from fastapi import APIRouter

from riskloom.api.routes import checkout, dashboard, health, webhooks

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(webhooks.router)
api_router.include_router(checkout.router)
api_router.include_router(dashboard.router)
