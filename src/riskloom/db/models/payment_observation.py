from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from riskloom.db.base import Base

if TYPE_CHECKING:
    from riskloom.db.models.webhook_event import WebhookEvent


class PaymentObservation(Base):
    __tablename__ = "payment_observations"
    __table_args__ = (
        CheckConstraint("amount >= 0", name="amount_non_negative"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_iso_shape"),
        Index("ix_payment_observations_provider_payment_id", "provider_payment_id"),
        Index("ix_payment_observations_provider_order_id", "provider_order_id"),
        Index("ix_payment_observations_provider_event_created_at", "provider_event_created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    webhook_event_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("webhook_events.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    provider_payment_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_order_id: Mapped[str | None] = mapped_column(String(255))
    event_name: Mapped[str] = mapped_column(String(100), nullable=False)
    payment_status: Mapped[str] = mapped_column(String(50), nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    payment_method: Mapped[str | None] = mapped_column(String(50))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_source: Mapped[str | None] = mapped_column(String(100))
    error_step: Mapped[str | None] = mapped_column(String(100))
    error_reason: Mapped[str | None] = mapped_column(String(100))
    provider_event_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    provider_payment_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    webhook_event: Mapped["WebhookEvent"] = relationship(back_populates="payment_observation")
