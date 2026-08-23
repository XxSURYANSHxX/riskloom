"""The locked PSI reference, read from the one-time held-out evaluation artifact.

This module reads ``artifacts/evaluations/development/evaluation.json`` and nothing else. The
protected test partition is never opened, re-scored, or re-derived: only the already-computed,
already-locked per-bin counts are used, exactly as published. The file is opened read-only and no
value is modified.

A caveat that belongs in the code rather than only in a report: the locked reference is heavily
degenerate for this purpose. Of 17,000 held-out rows, 97.78% fall in the first bin, five bins are
empty, and the locked decision threshold itself falls *inside* the first bin. PSI over this binning
therefore has almost no resolution where decisions actually happen, and a reading is dominated by
the high-probability bins that carry the evaluation set's attack traffic. The number is worth
watching for gross movement; it is not a precision instrument, and the surface says so.
"""

import json
from pathlib import Path

from riskloom.drift.schemas import ReferenceBin

EXPECTED_BIN_COUNT = 10


class ReferenceUnavailableError(RuntimeError):
    """The locked evaluation artifact is absent or unusable.

    The artifact tree is Git-ignored, so absence is an ordinary state for a fresh clone rather than
    an error, and the surface reports it plainly instead of failing.
    """


class LockedReference:
    """Immutable view of the locked bin distribution."""

    def __init__(
        self,
        bins: list[ReferenceBin],
        row_count: int,
        model_id: str | None,
        evaluation_id: str | None,
    ) -> None:
        self.bins = bins
        self.row_count = row_count
        self.model_id = model_id
        self.evaluation_id = evaluation_id

    @property
    def counts(self) -> list[int]:
        return [item.count for item in self.bins]

    @property
    def shares(self) -> list[float]:
        return [item.share for item in self.bins]

    def bin_index_for(self, probability: float) -> int:
        """Which locked bin a probability falls into.

        Uses the artifact's own edges rather than assuming uniform width, and honours the final
        bin's inclusive upper bound so a probability of exactly 1.0 is counted rather than dropped.
        """

        for item in self.bins:
            if item.upper_inclusive:
                if item.lower_inclusive <= probability <= item.upper_value:
                    return item.index
            elif item.lower_inclusive <= probability < item.upper_value:
                return item.index
        raise ValueError("drift_probability_outside_reference_range")


def load_reference(path: Path) -> LockedReference:
    """Read and validate the locked bins. Never writes, never mutates."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceUnavailableError("drift_reference_unreadable") from exc

    if not isinstance(payload, dict):
        raise ReferenceUnavailableError("drift_reference_invalid")

    metrics = payload.get("metrics")
    reliability = metrics.get("reliability") if isinstance(metrics, dict) else None
    raw_bins = reliability.get("bins") if isinstance(reliability, dict) else None
    if not isinstance(raw_bins, list) or len(raw_bins) != EXPECTED_BIN_COUNT:
        raise ReferenceUnavailableError("drift_reference_bins_invalid")

    row_count = payload.get("row_count")
    if not isinstance(row_count, int) or row_count <= 0:
        raise ReferenceUnavailableError("drift_reference_row_count_invalid")

    bins: list[ReferenceBin] = []
    for index, raw in enumerate(raw_bins):
        if not isinstance(raw, dict):
            raise ReferenceUnavailableError("drift_reference_bins_invalid")
        try:
            count = int(raw["count"])
            lower = float(raw["lower_inclusive"])
            upper = float(raw["upper_value"])
            upper_inclusive = bool(raw["upper_inclusive"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceUnavailableError("drift_reference_bins_invalid") from exc
        if count < 0 or upper <= lower:
            raise ReferenceUnavailableError("drift_reference_bins_invalid")
        bins.append(
            ReferenceBin(
                index=index,
                lower_inclusive=lower,
                upper_value=upper,
                upper_inclusive=upper_inclusive,
                count=count,
                share=count / row_count,
            )
        )

    # The published bins partition the evaluated population exactly. A mismatch means the artifact
    # is not the one this reference was designed against, and guessing would be worse than failing.
    if sum(item.count for item in bins) != row_count:
        raise ReferenceUnavailableError("drift_reference_counts_do_not_sum")

    model_id = payload.get("model_id")
    evaluation_id = payload.get("evaluation_id")
    return LockedReference(
        bins=bins,
        row_count=row_count,
        model_id=model_id if isinstance(model_id, str) else None,
        evaluation_id=evaluation_id if isinstance(evaluation_id, str) else None,
    )
