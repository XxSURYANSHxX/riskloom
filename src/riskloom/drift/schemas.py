"""Strict models for the drift surface.

Every field here is an aggregate or a bin edge. No pseudonymous token, identifier or per-event
value appears, and there is deliberately no field that could carry one.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from riskloom.drift.psi import DriftBand


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReferenceBin(_Model):
    """One locked bin, exactly as recorded in the held-out evaluation artifact."""

    index: int
    lower_inclusive: float
    upper_value: float
    upper_inclusive: bool
    count: int
    share: float


class DriftBinReport(_Model):
    """One bin's side-by-side comparison and its share of the total PSI."""

    index: int
    lower_inclusive: float
    upper_value: float
    reference_count: int
    reference_share: float
    observed_count: int
    observed_share: float
    contribution: float
    contribution_share: float


class DriftReport(_Model):
    """The complete drift answer.

    ``status`` is the honest gate: below the minimum row count there is no ``psi`` and no ``band``
    at all, rather than a number the sample cannot support.
    """

    status: Literal["ok", "insufficient_data", "reference_unavailable"]
    psi: float | None
    band: DriftBand | None
    observed_rows: int
    minimum_rows: int
    window_hours: int
    epsilon: float
    reference_rows: int
    reference_model_id: str | None
    reference_evaluation_id: str | None
    bins: list[DriftBinReport]
    note: str
