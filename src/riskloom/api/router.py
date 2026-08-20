from fastapi import APIRouter

from riskloom.api.routes import health, webhooks

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(webhooks.router)
