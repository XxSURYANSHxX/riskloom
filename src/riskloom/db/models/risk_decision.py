from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from riskloom.db.base import Base

if TYPE_CHECKING:
    from riskloom.db.models.review_item import ReviewItem


class RiskDecision(Base):
    """Append-only audit ledger for every live preflight decision.

    Only pseudonymous, model-visible fields are stored. There is deliberately no column for an
    email address, phone number, cardholder name, card number, VPA or network address, so no such
    value can be written here even by mistake. Rows transition once from ``pending`` to ``final``
    and are never deleted or re-decided.
    """

    __tablename__ = "risk_decisions"
    __table_args__ = (
        CheckConstraint("risk_decision IN ('allow', 'deny')", name="risk_decision_allowed"),
        CheckConstraint("action IN ('allow', 'review', 'deny')", name="action_allowed"),
        CheckConstraint("status IN ('pending', 'final')", name="status_allowed"),
        CheckConstraint("amount_subunits >= 1", name="amount_subunits_positive"),
        CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_iso_shape"),
        Index("ix_risk_decisions_merchant_id", "merchant_id"),
        Index("ix_risk_decisions_checkout_id", "checkout_id"),
        Index("ix_risk_decisions_occurred_at", "occurred_at"),
        Index("ix_risk_decisions_action", "action"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    checkout_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_token: Mapped[str] = mapped_column(String(64), nullable=False)
    payment_instrument_token: Mapped[str] = mapped_column(String(64), nullable=False)
    customer_token: Mapped[str | None] = mapped_column(String(64))
    device_token: Mapped[str | None] = mapped_column(String(64))
    network_token: Mapped[str | None] = mapped_column(String(64))

    amount_subunits: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)

    # Assigned by the feature engine under its lock, which runs only after the idempotency claim
    # has succeeded, so this is null while the row is `pending`.
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Storage and audit only. The live decision compares the full float64 probability against the
    # full float64 threshold loaded from model.json; neither value is ever read back through this
    # column to make a decision.
    calibrated_probability: Mapped[Decimal | None] = mapped_column(Numeric(20, 18))
    decision_threshold: Mapped[Decimal] = mapped_column(Numeric(20, 18), nullable=False)

    risk_decision: Mapped[str | None] = mapped_column(String(16))
    action: Mapped[str | None] = mapped_column(String(16))
    fail_safe_reason: Mapped[str | None] = mapped_column(String(64))

    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(16), nullable=False)
    feature_engine_version: Mapped[str] = mapped_column(String(16), nullable=False)

    razorpay_order_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    review_item: Mapped["ReviewItem | None"] = relationship(
        back_populates="risk_decision", uselist=False
    )
