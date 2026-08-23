"""Reading the locked PSI reference out of the held-out evaluation artifact.

The artifact is the single source for the reference distribution. These tests assert it is read
faithfully and left untouched, and they record the properties that make it a weak reference so a
future reader is not misled by a tidy-looking number.
"""

import hashlib
import json
from pathlib import Path

import pytest

from riskloom.drift.reference import (
    EXPECTED_BIN_COUNT,
    ReferenceUnavailableError,
    load_reference,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LOCKED = REPOSITORY_ROOT / "artifacts" / "evaluations" / "development" / "evaluation.json"

locked_only = pytest.mark.skipif(
    not LOCKED.exists(), reason="locked evaluation artifact is Git-ignored and absent"
)


def write_artifact(path: Path, bins: list[dict], row_count: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "row_count": row_count,
                "model_id": "m" * 64,
                "evaluation_id": "e" * 64,
                "metrics": {"reliability": {"bins": bins}},
            }
        ),
        encoding="utf-8",
    )
    return path


def uniform_bins(counts: list[int]) -> list[dict]:
    width = 1.0 / len(counts)
    return [
        {
            "count": count,
            "lower_inclusive": index * width,
            "upper_value": (index + 1) * width,
            "upper_inclusive": index == len(counts) - 1,
        }
        for index, count in enumerate(counts)
    ]


# --------------------------------------------------------------------------- the real artifact


@locked_only
def test_the_locked_bins_load_without_modification() -> None:
    before = hashlib.sha256(LOCKED.read_bytes()).hexdigest()
    reference = load_reference(LOCKED)
    after = hashlib.sha256(LOCKED.read_bytes()).hexdigest()

    assert before == after, "reading the reference must never rewrite the artifact"
    assert len(reference.bins) == EXPECTED_BIN_COUNT
    assert sum(reference.counts) == reference.row_count
    assert sum(reference.shares) == pytest.approx(1.0, abs=1e-12)


@locked_only
def test_the_locked_reference_is_recorded_as_degenerate() -> None:
    """Not a defect to fix, a property to remember.

    The reference concentrates almost everything in the first bin and leaves five bins empty, which
    is why the epsilon floor exists and why the surface refuses to report a band from a small
    sample. If this ever stops being true the comment in ``reference.py`` needs revisiting.
    """

    reference = load_reference(LOCKED)
    shares = reference.shares
    assert shares[0] > 0.97
    assert sum(1 for count in reference.counts if count == 0) == 5

    # The locked decision threshold falls inside the first bin, so the binning has no resolution
    # where decisions are actually made.
    threshold = 0.0033862949155182734
    assert reference.bin_index_for(threshold) == 0


@locked_only
def test_bin_lookup_covers_the_whole_unit_interval() -> None:
    reference = load_reference(LOCKED)
    assert reference.bin_index_for(0.0) == 0
    assert reference.bin_index_for(0.05) == 0
    assert reference.bin_index_for(0.35) == 3
    assert reference.bin_index_for(0.95) == 9
    # The final bin is inclusive at the top, so a probability of exactly 1.0 is counted.
    assert reference.bin_index_for(1.0) == 9


# --------------------------------------------------------------------------- validation


def test_a_missing_artifact_is_an_ordinary_state(tmp_path: Path) -> None:
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_unreadable"):
        load_reference(tmp_path / "absent.json")


def test_unparseable_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_unreadable"):
        load_reference(path)


def test_the_wrong_number_of_bins_is_refused(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "few.json", uniform_bins([10, 10]), 20)
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_bins_invalid"):
        load_reference(path)


def test_counts_that_do_not_sum_to_the_row_count_are_refused(tmp_path: Path) -> None:
    """A mismatch means this is not the artifact the reference was designed against."""

    path = write_artifact(tmp_path / "mismatch.json", uniform_bins([1] * 10), 999)
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_counts_do_not_sum"):
        load_reference(path)


def test_a_negative_count_is_refused(tmp_path: Path) -> None:
    bins = uniform_bins([1] * 10)
    bins[3]["count"] = -1
    path = write_artifact(tmp_path / "negative.json", bins, 9)
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_bins_invalid"):
        load_reference(path)


def test_inverted_bin_edges_are_refused(tmp_path: Path) -> None:
    bins = uniform_bins([1] * 10)
    bins[2]["upper_value"] = bins[2]["lower_inclusive"]
    path = write_artifact(tmp_path / "inverted.json", bins, 10)
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_bins_invalid"):
        load_reference(path)


def test_a_missing_row_count_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "norows.json"
    path.write_text(
        json.dumps({"metrics": {"reliability": {"bins": uniform_bins([1] * 10)}}}),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceUnavailableError, match="drift_reference_row_count_invalid"):
        load_reference(path)


def test_a_probability_outside_the_reference_range_is_refused(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "ok.json", uniform_bins([1] * 10), 10)
    reference = load_reference(path)
    with pytest.raises(ValueError, match="drift_probability_outside_reference_range"):
        reference.bin_index_for(1.5)
