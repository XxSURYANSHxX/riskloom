from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from riskloom.db.base import Base

if TYPE_CHECKING:
    from riskloom.db.models.risk_decision import RiskDecision


class ReviewItem(Base):
    """A checkout held for manual review.

    Day 6 only records these. There is deliberately no resolution column, no reviewer column and
    no auto-resolution logic: working the queue is a later gate.
    """

    __tablename__ = "review_items"
    __table_args__ = (
        CheckConstraint("status = 'pending'", name="status_allowed"),
        Index("ix_review_items_merchant_id", "merchant_id"),
        Index("ix_review_items_created_at", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    risk_decision_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("risk_decisions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    checkout_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    risk_decision: Mapped["RiskDecision"] = relationship(back_populates="review_item")
