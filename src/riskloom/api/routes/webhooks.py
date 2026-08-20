from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.api.dependencies import get_app_settings, get_session
from riskloom.core.config import Settings
from riskloom.integrations.razorpay.schemas import WebhookEnvelope
from riskloom.integrations.razorpay.signatures import sha256_digest, verify_webhook_signature
from riskloom.services.webhook_ingestion import WebhookPayloadError, ingest_webhook

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])
logger = structlog.get_logger("riskloom.webhooks")


def _error(request: Request, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "request_id": request.state.request_id}},
    )


async def _read_bounded_body(request: Request, maximum_bytes: int) -> bytes | None:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > maximum_bytes:
                return None
        except ValueError:
            return None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > maximum_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/razorpay")
async def razorpay_webhook(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JSONResponse:
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().casefold()
    if media_type != "application/json":
        return _error(request, 415, "unsupported_content_type")

    raw_body = await _read_bounded_body(request, settings.webhook_max_body_bytes)
    if raw_body is None:
        return _error(request, 413, "request_too_large")

    received_signature = request.headers.get("x-razorpay-signature")
    if received_signature is None:
        logger.warning("webhook_rejected", reason="missing_signature")
        return _error(request, 401, "missing_signature")
    if not verify_webhook_signature(
        raw_body,
        received_signature,
        settings.razorpay_webhook_secret,
    ):
        logger.warning("webhook_rejected", reason="invalid_signature")
        return _error(request, 401, "invalid_signature")

    provider_event_id = request.headers.get("x-razorpay-event-id")
    if provider_event_id is None or not 1 <= len(provider_event_id) <= 255:
        return _error(request, 400, "invalid_event_id")

    try:
        envelope = WebhookEnvelope.model_validate_json(raw_body)
    except ValueError:
        return _error(request, 400, "invalid_webhook_envelope")

    digest = sha256_digest(raw_body)
    try:
        result = await ingest_webhook(
            session,
            provider_event_id=provider_event_id,
            envelope=envelope,
            raw_body_sha256=digest,
        )
    except WebhookPayloadError:
        return _error(request, 400, "invalid_payment_payload")
    except SQLAlchemyError:
        logger.error("webhook_storage_failed", reason="database_error")
        return _error(request, 503, "storage_unavailable")
    except Exception:
        logger.error("webhook_processing_failed", reason="internal_error")
        return _error(request, 500, "internal_error")

    logger.info(
        "webhook_accepted",
        event_name=envelope.event,
        duplicate=result.duplicate,
    )
    return JSONResponse(
        status_code=200,
        content={"status": "accepted", "duplicate": result.duplicate},
    )
