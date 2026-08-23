"""Read-only drift comparison over the live ledger.

Every database interaction in the drift feature lives here, and every one of them is a ``SELECT``.
The drift package itself holds no session and imports no ORM, so the arithmetic side of this
feature cannot write to any table even by mistake.

Nothing here influences a decision. It reads `risk_decisions` after the fact and returns a number.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from riskloom.db.models import RiskDecision
from riskloom.drift import psi as psi_module
from riskloom.drift.reference import LockedReference, ReferenceUnavailableError, load_reference
from riskloom.drift.schemas import DriftBinReport, DriftReport

DEFAULT_WINDOW_HOURS = 24
MAXIMUM_WINDOW_HOURS = 24 * 30
MAXIMUM_ROWS = 50_000

INFORMATIONAL_NOTE = (
    "Informational only. Drift is computed after the fact from stored scores and has no path to "
    "any decision, threshold, or model parameter."
)
INSUFFICIENT_NOTE = (
    "Too few scored decisions in this window to report a stability index. A band computed from a "
    "handful of rows would be noise, so none is shown."
)
UNAVAILABLE_NOTE = (
    "The locked held-out evaluation artifact is not present, so there is no reference "
    "distribution to compare against."
)


async def observed_probabilities(
    session: AsyncSession, window_hours: int, limit: int = MAXIMUM_ROWS
) -> list[float]:
    """Calibrated probabilities of recent scored decisions. Read-only.

    A time window rather than a fixed row count: an operator reasons in hours, and 'the last N
    rows' can silently reach back weeks on a quiet ledger and present stale traffic as current.
    """

    anchor = func.coalesce(RiskDecision.occurred_at, RiskDecision.created_at)
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    rows = await session.scalars(
        select(RiskDecision.calibrated_probability)
        .where(anchor >= since)
        .where(RiskDecision.calibrated_probability.is_not(None))
        .order_by(anchor.desc())
        .limit(limit)
    )
    return [float(value) for value in rows if isinstance(value, Decimal)]


def _binned(reference: LockedReference, probabilities: list[float]) -> list[int]:
    counts = [0] * len(reference.bins)
    for probability in probabilities:
        # Probabilities are calibrated to [0, 1]; clamp defensively so a stored value at a float
        # edge cannot fall outside every bin and be silently dropped from the comparison.
        bounded = min(max(probability, 0.0), 1.0)
        counts[reference.bin_index_for(bounded)] += 1
    return counts


def build_report(
    reference: LockedReference, probabilities: list[float], window_hours: int
) -> DriftReport:
    """Compare an observed sample against the locked reference."""

    observed_rows = len(probabilities)
    counts = _binned(reference, probabilities) if observed_rows else [0] * len(reference.bins)

    if observed_rows < psi_module.PSI_MINIMUM_ROWS:
        return DriftReport(
            status="insufficient_data",
            psi=None,
            band=None,
            observed_rows=observed_rows,
            minimum_rows=psi_module.PSI_MINIMUM_ROWS,
            window_hours=window_hours,
            epsilon=psi_module.PSI_EPSILON,
            reference_rows=reference.row_count,
            reference_model_id=reference.model_id,
            reference_evaluation_id=reference.evaluation_id,
            bins=_bin_reports(reference, counts, observed_rows, None),
            note=f"{INSUFFICIENT_NOTE} {INFORMATIONAL_NOTE}",
        )

    value, terms = psi_module.from_counts(reference.counts, counts)
    return DriftReport(
        status="ok",
        psi=value,
        band=psi_module.band(value),
        observed_rows=observed_rows,
        minimum_rows=psi_module.PSI_MINIMUM_ROWS,
        window_hours=window_hours,
        epsilon=psi_module.PSI_EPSILON,
        reference_rows=reference.row_count,
        reference_model_id=reference.model_id,
        reference_evaluation_id=reference.evaluation_id,
        bins=_bin_reports(reference, counts, observed_rows, terms),
        note=INFORMATIONAL_NOTE,
    )


def _bin_reports(
    reference: LockedReference,
    counts: list[int],
    observed_rows: int,
    terms: list[float] | None,
) -> list[DriftBinReport]:
    total_contribution = sum(abs(term) for term in terms) if terms else 0.0
    reports: list[DriftBinReport] = []
    for item, observed in zip(reference.bins, counts, strict=True):
        contribution = terms[item.index] if terms else 0.0
        reports.append(
            DriftBinReport(
                index=item.index,
                lower_inclusive=item.lower_inclusive,
                upper_value=item.upper_value,
                reference_count=item.count,
                reference_share=item.share,
                observed_count=observed,
                observed_share=(observed / observed_rows) if observed_rows else 0.0,
                contribution=contribution,
                contribution_share=(
                    abs(contribution) / total_contribution if total_contribution > 0 else 0.0
                ),
            )
        )
    return reports


async def evaluate_drift(
    session: AsyncSession, reference_path: Path, window_hours: int = DEFAULT_WINDOW_HOURS
) -> DriftReport:
    """Full drift answer, degrading plainly when the locked reference is absent."""

    window = max(1, min(window_hours, MAXIMUM_WINDOW_HOURS))
    try:
        reference = load_reference(reference_path)
    except ReferenceUnavailableError:
        return DriftReport(
            status="reference_unavailable",
            psi=None,
            band=None,
            observed_rows=0,
            minimum_rows=psi_module.PSI_MINIMUM_ROWS,
            window_hours=window,
            epsilon=psi_module.PSI_EPSILON,
            reference_rows=0,
            reference_model_id=None,
            reference_evaluation_id=None,
            bins=[],
            note=f"{UNAVAILABLE_NOTE} {INFORMATIONAL_NOTE}",
        )

    probabilities = await observed_probabilities(session, window)
    return build_report(reference, probabilities, window)
