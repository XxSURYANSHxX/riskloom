from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.db.models import PaymentObservation, WebhookEvent
from riskloom.integrations.razorpay.sanitizer import sanitize_webhook_payload
from riskloom.integrations.razorpay.schemas import PaymentEntity, WebhookEnvelope

SUPPORTED_PAYMENT_EVENTS = frozenset({"payment.authorized", "payment.captured", "payment.failed"})


class WebhookPayloadError(Exception):
    """The signed webhook does not contain the documented fields required for processing."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    duplicate: bool


def _payment_from_envelope(envelope: WebhookEnvelope) -> PaymentEntity | None:
    if envelope.event not in SUPPORTED_PAYMENT_EVENTS:
        return None
    try:
        payment_wrapper = envelope.payload["payment"]
        if not isinstance(payment_wrapper, dict):
            raise TypeError
        payment_entity = payment_wrapper["entity"]
        return PaymentEntity.model_validate(payment_entity)
    except (KeyError, TypeError, ValidationError) as exc:
        raise WebhookPayloadError("invalid_supported_payment_payload") from exc


async def ingest_webhook(
    session: AsyncSession,
    *,
    provider_event_id: str,
    envelope: WebhookEnvelope,
    raw_body_sha256: str,
) -> IngestionResult:
    payment = _payment_from_envelope(envelope)
    processing_result = "processed" if payment is not None else "ignored"
    sanitized_payload: dict[str, Any] = sanitize_webhook_payload(envelope.model_dump(mode="python"))
    provider_created_at = datetime.fromtimestamp(envelope.created_at, tz=UTC)

    async with session.begin():
        event_id = await session.scalar(
            insert(WebhookEvent)
            .values(
                provider_event_id=provider_event_id,
                event_name=envelope.event,
                provider_created_at=provider_created_at,
                raw_body_sha256=raw_body_sha256,
                sanitized_payload=sanitized_payload,
                processing_result=processing_result,
            )
            .on_conflict_do_nothing(index_elements=[WebhookEvent.provider_event_id])
            .returning(WebhookEvent.id)
        )
        if event_id is None:
            return IngestionResult(duplicate=True)

        if payment is not None:
            await session.execute(
                insert(PaymentObservation).values(
                    webhook_event_id=event_id,
                    provider_payment_id=payment.id,
                    provider_order_id=payment.order_id,
                    event_name=envelope.event,
                    payment_status=payment.status,
                    amount=payment.amount,
                    currency=payment.currency,
                    payment_method=payment.method,
                    error_code=payment.error_code,
                    error_source=payment.error_source,
                    error_step=payment.error_step,
                    error_reason=payment.error_reason,
                    provider_event_created_at=provider_created_at,
                    provider_payment_created_at=datetime.fromtimestamp(payment.created_at, tz=UTC),
                )
            )

    return IngestionResult(duplicate=False)
