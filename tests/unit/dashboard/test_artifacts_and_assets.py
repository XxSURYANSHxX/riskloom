"""The offline artifact projection, and that the shipped client is wired to what exists."""

import json
import re
from pathlib import Path

import pytest

from riskloom.dashboard.artifacts import EvaluationUnavailableError, load_model_evaluation

STATIC = Path("static")
REAL_ARTIFACT = Path("artifacts/evaluations/development/evaluation.json")


def _minimal() -> dict:
    return {
        "model_id": "a" * 64,
        "evaluation_id": "b" * 64,
        "row_count": 10,
        "metrics": {
            "threshold": {
                "row_count": 10,
                "attack_count": 2,
                "legitimate_count": 8,
                "true_positive": 1,
                "false_positive": 1,
                "true_negative": 7,
                "false_negative": 1,
                "precision": 0.5,
                "recall": 0.5,
                "specificity": 0.875,
                "f1_score": 0.5,
                "false_positive_rate": 0.125,
                "false_positives_per_10000_legitimate": 1250.0,
                "prevalence": 0.2,
                "review_count": 2,
                "review_rate": 0.2,
                "cost_units": 26,
                "cost_per_10000_events": 26000.0,
                "threshold": 0.5,
            },
            "probability": {
                "average_precision": 0.4,
                "pr_auc_trapezoidal": 0.5,
                "roc_auc": 0.6,
                "brier_score": 0.1,
                "log_loss": 0.2,
            },
            "reliability": {
                "expected_calibration_error": 0.01,
                "bins": [
                    {
                        "lower_inclusive": 0.0,
                        "upper_value": 1.0,
                        "upper_inclusive": True,
                        "count": 10,
                        "mean_probability": 0.3,
                        "attack_rate": 0.2,
                    }
                ],
            },
            "hard_negative_slices": {
                "normal": {
                    "row_count": 8,
                    "false_positive_count": 1,
                    "false_positive_rate": 0.125,
                    "false_positives_per_10000": 1250.0,
                }
            },
            "campaigns": {
                "campaign_count": 1,
                "detected_campaign_count": 1,
                "missed_campaign_count": 0,
                "campaign_recall": 1.0,
                "detection_delay_ms": {"minimum": 1, "median": 2, "p95": 3, "maximum": 4},
                "flagged_events_per_campaign": {
                    "minimum": 1,
                    "median": 2,
                    "maximum": 3,
                    "mean": 2.0,
                    "p95": 3,
                    "histogram": {"1": 1},
                },
            },
        },
    }


def test_absent_artifact_is_an_ordinary_outcome(tmp_path: Path) -> None:
    with pytest.raises(EvaluationUnavailableError, match="absent"):
        load_model_evaluation(tmp_path / "nothing.json")


def test_unreadable_and_malformed_artifacts_are_refused(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_bytes(b"not json")
    with pytest.raises(EvaluationUnavailableError, match="unreadable"):
        load_model_evaluation(broken)

    partial = tmp_path / "partial.json"
    partial.write_bytes(json.dumps({"model_id": "x"}).encode())
    with pytest.raises(EvaluationUnavailableError, match="malformed"):
        load_model_evaluation(partial)


def test_projection_selects_fields_by_name(tmp_path: Path) -> None:
    """A per-event section in the source must not survive the projection."""

    source = _minimal()
    source["per_event_predictions"] = [{"event_id": "evt_" + "1" * 32, "probability": 0.9}]
    source["metrics"]["campaigns"]["campaign_ids"] = ["cmp_" + "2" * 32]
    path = tmp_path / "evaluation.json"
    path.write_bytes(json.dumps(source).encode())

    projected = load_model_evaluation(path).model_dump(mode="json")
    rendered = json.dumps(projected)
    assert "per_event_predictions" not in rendered
    assert "campaign_ids" not in rendered
    assert "evt_" not in rendered
    assert "cmp_" not in rendered
    # The histogram, which is keyed by flagged-event count, is also not carried through.
    assert "histogram" not in rendered


def test_projection_preserves_the_values_it_does_carry(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    path.write_bytes(json.dumps(_minimal()).encode())
    projected = load_model_evaluation(path)
    assert projected.threshold.recall == 0.5
    assert projected.campaigns.campaign_recall == 1.0
    assert projected.campaigns.detection_delay_ms_p95 == 3
    assert projected.hard_negative_slices[0].slice_name == "normal"
    assert len(projected.reliability.bins) == 1


@pytest.mark.skipif(not REAL_ARTIFACT.exists(), reason="artifact is Git-ignored")
def test_real_artifact_projects_cleanly() -> None:
    projected = load_model_evaluation(REAL_ARTIFACT)
    assert len(projected.reliability.bins) == 10
    assert len(projected.hard_negative_slices) == 5
    assert projected.campaigns.campaign_count == 3


# ------------------------------------------------------------------- assets


def test_every_asset_referenced_by_the_page_exists() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for reference in re.findall(r'(?:href|src)="\./([^"]+)"', html):
        assert (STATIC / reference).exists(), reference


def test_stylesheet_fonts_are_vendored_not_fetched() -> None:
    """A demo must not depend on a font CDN being reachable."""

    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for reference in re.findall(r'url\("\./([^"]+)"\)', css):
        asset = STATIC / reference
        assert asset.exists(), reference
        assert asset.read_bytes()[:4] == b"wOF2", reference
    assert "https://" not in css
    assert "fonts.googleapis.com" not in css


# The SVG XML namespace is an identifier, not a fetched origin, and is the one legitimate
# occurrence of a URL-shaped literal in the client.
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def test_client_modules_resolve_and_declare_no_external_origin() -> None:
    """Every import resolves locally and nothing is fetched from another origin."""

    for name in sorted(path.name for path in STATIC.glob("*.js")):
        source = (STATIC / name).read_text(encoding="utf-8")
        for reference in re.findall(r'from "\./([^"]+)"', source):
            assert (STATIC / reference).exists(), reference
        remaining = source.replace(SVG_NAMESPACE, "")
        assert "http://" not in remaining, name
        assert "https://" not in remaining, name
        # Every fetch target is a site-relative path.
        for target in re.findall(r"fetch\(([^)]*)\)", source):
            assert "//" not in target, (name, target)


def test_client_only_ever_calls_the_read_only_dashboard_api() -> None:
    """No mutation path: the client issues no method other than the default GET."""

    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "method:" not in source
    for verb in ('"POST"', '"PUT"', '"PATCH"', '"DELETE"'):
        assert verb not in source
    for path in re.findall(r"\$\{API\}(/[a-z/{}$.\w-]*)", source):
        assert path.startswith("/")
    assert '"/api/v1/dashboard"' in source or "'/api/v1/dashboard'" in source


def test_client_never_references_a_policy_band_or_a_feature_name() -> None:
    """The dashboard displays stored data; it must not imply model internals it cannot see."""

    for name in sorted(path.name for path in STATIC.glob("*.js")):
        source = (STATIC / name).read_text(encoding="utf-8").casefold()
        for token in ("policy_band", "select_band", "prior_attempt_count", "feature_vector"):
            assert token not in source, (name, token)
