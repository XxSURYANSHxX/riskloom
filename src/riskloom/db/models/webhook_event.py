from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from riskloom.db.base import Base

if TYPE_CHECKING:
    from riskloom.db.models.payment_observation import PaymentObservation


class WebhookEvent(Base):
    __tablename__ = "webhook_events"
    __table_args__ = (
        CheckConstraint(
            "processing_result IN ('processed', 'ignored')",
            name="processing_result_allowed",
        ),
        Index("ix_webhook_events_event_name", "event_name"),
        Index("ix_webhook_events_provider_created_at", "provider_created_at"),
        Index("ix_webhook_events_received_at", "received_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    event_name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    raw_body_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    sanitized_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processing_result: Mapped[str] = mapped_column(String(16), nullable=False)

    payment_observation: Mapped["PaymentObservation | None"] = relationship(
        back_populates="webhook_event",
        uselist=False,
    )
