from fastapi import APIRouter

from riskloom.api.routes import checkout, dashboard, drift, explanations, health, webhooks

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(webhooks.router)
api_router.include_router(checkout.router)
api_router.include_router(dashboard.router)
api_router.include_router(explanations.router)
api_router.include_router(drift.router)
