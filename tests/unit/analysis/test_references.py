"""Reference rows must come from their source artifacts, never from literals.

A hardcoded baseline is a number that quietly stops being true. These tests re-read the sources
independently and compare exactly, and then scan the analysis package for any constant that looks
like a reference figure -- which is how a literal usually reappears: not in the first version of a
module, but in an edit months later.
"""

import ast
import json
from pathlib import Path

import pytest

from riskloom.analysis.references import (
    EVALUATION_PATH,
    POLICY_COMPARISON_PATH,
    ReferenceError,
    load_held_out_reference,
    load_policy_validation_reference,
    load_references,
)

ANALYSIS_PACKAGE = Path("src/riskloom/analysis")

held_out_only = pytest.mark.skipif(
    not EVALUATION_PATH.exists(), reason="locked evaluation artifact absent"
)
policy_only = pytest.mark.skipif(
    not POLICY_COMPARISON_PATH.exists(), reason="policy comparison artifact absent"
)


# --------------------------------------------------------------------------- read from source


@held_out_only
def test_the_held_out_row_matches_the_artifact_exactly() -> None:
    source = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))["metrics"]
    row = load_held_out_reference()

    assert row.available
    assert row.recall == source["threshold"]["recall"]
    assert row.precision == source["threshold"]["precision"]
    assert row.false_positive_rate == source["threshold"]["false_positive_rate"]
    assert row.cost_units == source["threshold"]["cost_units"]
    assert row.average_precision == source["probability"]["average_precision"]
    assert row.row_count == source["threshold"]["row_count"]


@policy_only
def test_the_policy_row_is_derived_from_the_artifact_counts() -> None:
    outcome = json.loads(POLICY_COMPARISON_PATH.read_text(encoding="utf-8"))["incumbent_policy"][
        "outcome"
    ]
    row = load_policy_validation_reference()

    assert row.available
    assert row.recall == outcome["true_positive"] / outcome["attack_count"]
    assert row.precision == outcome["true_positive"] / (
        outcome["true_positive"] + outcome["false_positive"]
    )
    assert row.false_positive_rate == outcome["false_positive_rate"]
    assert row.cost_units == outcome["cost_units"]


@policy_only
def test_the_policy_row_reports_no_average_precision() -> None:
    """It is genuinely absent from that artifact.

    The 0.4384 in the README came from a separate blindspot measurement. Carrying it here would be
    exactly the hardcoding this module exists to prevent, so the cell is null instead.
    """

    assert load_policy_validation_reference().average_precision is None


# --------------------------------------------------------------------------- no literals


REFERENCE_FIGURES = (
    0.9764705882352941,
    0.7685185185185185,
    0.9640050726365282,
    0.006002400960384154,
    0.8555555555555555,
    0.5767790262172284,
    0.012811791383219955,
    0.4384,
    300,
    763,
    154,
    113,
    17_000,
    9_000,
    340,
    180,
)


def test_no_reference_figure_is_hardcoded_anywhere_in_the_analysis_package() -> None:
    """Scans constants rather than text, so a figure inside a docstring does not trip it."""

    for path in sorted(ANALYSIS_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, int | float)
        }
        for figure in REFERENCE_FIGURES:
            assert figure not in constants, (path.name, figure)


def test_the_loader_holds_no_numeric_constant_at_all() -> None:
    """references.py should contain only tolerances and structural zeros."""

    tree = ast.parse((ANALYSIS_PACKAGE / "references.py").read_text(encoding="utf-8"))
    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float)
    }
    assert numbers <= {0, 1, 1e-9}, numbers


# --------------------------------------------------------------------------- degradation


def test_a_missing_artifact_reports_unavailable_rather_than_a_fallback(tmp_path: Path) -> None:
    absent = tmp_path / "absent.json"
    for row in (load_held_out_reference(absent), load_policy_validation_reference(absent)):
        assert row.available is False
        assert row.recall is None
        assert row.cost_units is None
        assert "not present" in row.note


def test_unparseable_artifacts_report_unavailable(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_held_out_reference(broken).available is False
    assert load_policy_validation_reference(broken).available is False


# --------------------------------------------------------------------------- consistency


def test_a_doctored_policy_artifact_is_rejected(tmp_path: Path) -> None:
    """Counts that contradict the artifact's own summary must fail, not produce a plausible row."""

    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            {
                "incumbent_policy": {
                    "outcome": {
                        "true_positive": 154,
                        "false_negative": 26,
                        "false_positive": 113,
                        "legitimate_count": 8820,
                        "attack_count": 180,
                        "row_count": 9000,
                        "cost_units": 763,
                        # Inconsistent with 113 / 8820.
                        "false_positive_rate": 0.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceError, match="false_positive_rate_inconsistent"):
        load_policy_validation_reference(path)


def test_inconsistent_attack_counts_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            {
                "incumbent_policy": {
                    "outcome": {
                        "true_positive": 154,
                        "false_negative": 26,
                        "false_positive": 113,
                        "legitimate_count": 8820,
                        "attack_count": 999,
                        "row_count": 9000,
                        "cost_units": 763,
                        "false_positive_rate": 113 / 8820,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReferenceError, match="attack_counts_inconsistent"):
        load_policy_validation_reference(path)


def test_a_missing_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"metrics": {"threshold": {}}}), encoding="utf-8")
    with pytest.raises(ReferenceError, match="held_out_reference_missing_sections"):
        load_held_out_reference(path)


def test_load_references_returns_both_rows() -> None:
    rows = load_references()
    assert [row.label for row in rows] == ["Gate B2 held-out", "Gate C1 policy-validation"]
