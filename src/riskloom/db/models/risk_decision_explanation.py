from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from riskloom.db.base import Base

if TYPE_CHECKING:
    from riskloom.db.models.risk_decision import RiskDecision


class RiskDecisionExplanation(Base):
    """A generated explanation of an already-final decision.

    Enrichment, never a decision input. Nothing here can alter ``risk_decisions``: this table only
    references it, and the generation package that produces the content cannot reach a database at
    all.

    One row per attempt. A row makes a single transition from ``pending`` to a terminal status,
    the same discipline ``risk_decisions`` uses for ``pending`` to ``final``, and is never
    re-decided or deleted. A crashed run therefore leaves a ``pending`` row and is not
    auto-recovered.

    There is deliberately no column capable of holding an upstream response body. ``failure_reason``
    is a short stable identity only.
    """

    __tablename__ = "risk_decision_explanations"
    __table_args__ = (
        UniqueConstraint("risk_decision_id", "attempt_number", name="explanation_attempt_unique"),
        CheckConstraint(
            "status IN ('pending', 'ready', 'failed', 'rejected')", name="status_allowed"
        ),
        CheckConstraint("attempt_number >= 1", name="attempt_number_positive"),
        CheckConstraint("status <> 'ready' OR summary IS NOT NULL", name="ready_requires_summary"),
        CheckConstraint(
            "status NOT IN ('failed', 'rejected') OR failure_reason IS NOT NULL",
            name="terminal_failure_requires_reason",
        ),
        Index("ix_risk_decision_explanations_risk_decision_id", "risk_decision_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    risk_decision_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("risk_decisions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)

    summary: Mapped[str | None] = mapped_column(String(400))
    factors: Mapped[Any | None] = mapped_column(JSONB)
    caveat: Mapped[str | None] = mapped_column(String(300))
    failure_reason: Mapped[str | None] = mapped_column(String(64))

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(16), nullable=False)
    # SHA-256 of the exact canonical payload sent upstream. Proves after the fact which facts were
    # used, and lets a reviewer detect that the underlying ledger context has since changed.
    input_digest: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    risk_decision: Mapped["RiskDecision"] = relationship()
