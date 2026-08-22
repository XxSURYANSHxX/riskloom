"""Quantify the live-serving failure blind spot.

At preflight an attempt's outcome does not exist yet, so the online adapter advances feature state
with every attempt recorded as authorized. Eighteen of the seventy-five features are derived from
observed failures, so those read low for live-only traffic.

This module measures the cost of that assumption. It extracts features from one event stream twice
through the *same* unmodified ``FeatureEngine``:

* ``true_outcomes``  -- outcomes exactly as recorded, which is what offline training saw;
* ``assumed_authorized`` -- every state-advancing outcome forced to authorized, which is what live
  serving assumes.

It is a separate, explicitly named path. It does not change ``extract_feature_dataset`` and is
never used to produce a training artifact.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from riskloom.features.config import FeatureConfig
from riskloom.features.engine import FeatureEngine
from riskloom.features.schema import FEATURE_NAMES
from riskloom.simulation.event_schema import CheckoutAttemptEvent, Outcome

TRUE_OUTCOMES = "true_outcomes"
ASSUMED_AUTHORIZED = "assumed_authorized"


@dataclass(frozen=True, slots=True)
class ExtractionPair:
    feature_matrix: np.ndarray
    targets: np.ndarray


def _assume_authorized(event: CheckoutAttemptEvent) -> CheckoutAttemptEvent:
    """The live-serving assumption, applied offline.

    Only the outcome pair changes. Every other model-visible field, and the event's position in
    the stream, is identical.
    """

    if event.outcome is Outcome.AUTHORIZED:
        return event
    return event.model_copy(update={"outcome": Outcome.AUTHORIZED, "failure_category": None})


def extract(
    events_path: Path,
    labels_path: Path,
    config: FeatureConfig,
    *,
    mode: str,
) -> ExtractionPair:
    """Run one event stream through the unmodified engine under the requested outcome mode."""

    if mode not in (TRUE_OUTCOMES, ASSUMED_AUTHORIZED):
        raise ValueError("unsupported blind-spot extraction mode")

    engine = FeatureEngine(config)
    rows: list[list[int]] = []
    targets: list[int] = []
    with events_path.open("rb") as event_stream, labels_path.open("rb") as label_stream:
        for event_line, label_line in zip(event_stream, label_stream, strict=True):
            event = CheckoutAttemptEvent.model_validate_json(event_line)
            label: dict[str, Any] = json.loads(label_line)
            if label["event_id"] != event.event_id:
                raise ValueError("event and label streams are not aligned")
            advancing = event if mode == TRUE_OUTCOMES else _assume_authorized(event)
            record = engine.process(advancing)
            values = record.features.model_dump()
            rows.append([values[name] for name in FEATURE_NAMES])
            targets.append(int(label["is_attack"]))
    return ExtractionPair(
        feature_matrix=np.asarray(rows, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.int8),
    )
