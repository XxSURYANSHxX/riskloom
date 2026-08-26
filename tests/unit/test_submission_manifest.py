"""Evaluator-facing submission metadata stays strict, traceable, and internally consistent."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from riskloom.dashboard.artifacts import load_model_evaluation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "submission_manifest.json"
EVALUATOR_FILES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "JUDGING.md",
    MANIFEST_PATH,
)
REQUIRED_KEYS = {
    "schema_version",
    "project",
    "competition",
    "track",
    "problem",
    "solution",
    "baseline_release",
    "evaluator_readiness_package",
    "ai_judgment",
    "evaluation",
    "false_positive_cost",
    "safety_boundaries",
    "reproduction",
    "evidence",
    "known_limitations",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _manifest() -> dict[str, Any]:
    loaded = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite,
    )
    assert isinstance(loaded, dict)
    return loaded


def _referenced_paths(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == "path" or key.endswith("_path")) and isinstance(item, str):
                yield item
            yield from _referenced_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _referenced_paths(item)


def test_manifest_is_strict_json_with_the_required_contract() -> None:
    manifest = _manifest()
    assert manifest.keys() >= REQUIRED_KEYS
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["baseline_release"] == {
        "submission_tag": "v1.0.3-submission",
        "commit_sha": "ff36ba091f5bcf44b45e40044139996663f03bca",
        "runtime_release_tag": "v1.0.3-runtime",
        "immutable": True,
    }
    assert manifest["evaluator_readiness_package"] == {
        "based_on_tag": "v1.0.3-submission",
        "based_on_commit": "ff36ba091f5bcf44b45e40044139996663f03bca",
        "change_scope": [
            "documentation",
            "machine_readable_metadata",
            "validation_tests",
        ],
        "release_type": "source_submission",
        "model_changed": False,
        "runtime_changed": False,
        "runtime_bundle_changed": False,
        "artifact_hashes_changed": False,
        "features_changed": False,
        "apis_changed": False,
        "schema_changed": False,
        "dependencies_changed": False,
        "inference_behavior_changed": False,
    }


def test_every_referenced_evidence_path_is_safe_relative_and_exists() -> None:
    paths = list(_referenced_paths(_manifest()["evidence"]))
    paths.extend(_referenced_paths(_manifest()["evaluation"]))
    paths.extend(_referenced_paths(_manifest()["known_limitations"]))
    assert paths
    for relative in paths:
        path = PurePosixPath(relative)
        assert not path.is_absolute(), relative
        assert ".." not in path.parts, relative
        assert "\\" not in relative, relative
        assert (REPOSITORY_ROOT / Path(*path.parts)).exists(), relative


def test_locked_artifact_hashes_match_the_manifest() -> None:
    artifacts = _manifest()["evidence"]["locked_artifacts"]
    for artifact in artifacts:
        payload = (REPOSITORY_ROOT / artifact["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]


def test_critical_metrics_match_the_authoritative_evaluation_and_recompute() -> None:
    manifest = _manifest()
    declared = manifest["evaluation"]
    authoritative = load_model_evaluation(REPOSITORY_ROOT / declared["source"]["path"])
    threshold = authoritative.threshold
    campaigns = authoritative.campaigns

    assert declared["source"]["evaluation_id"] == authoritative.evaluation_id
    assert declared["row_count"] == authoritative.row_count == threshold.row_count
    assert declared["attack_count"] == threshold.attack_count
    assert declared["legitimate_count"] == threshold.legitimate_count
    assert declared["confusion_matrix"] == {
        "true_positive": threshold.true_positive,
        "false_positive": threshold.false_positive,
        "true_negative": threshold.true_negative,
        "false_negative": threshold.false_negative,
    }

    recall = declared["event_recall"]
    precision = declared["precision"]
    false_positive_rate = declared["false_positive_rate"]
    campaign_recall = declared["campaign_recall"]
    assert recall["numerator"] == threshold.true_positive
    assert recall["denominator"] == threshold.attack_count
    assert recall["value"] == pytest.approx(recall["numerator"] / recall["denominator"])
    assert recall["value"] == pytest.approx(threshold.recall)
    assert precision["numerator"] == threshold.true_positive
    assert precision["denominator"] == threshold.true_positive + threshold.false_positive
    assert precision["value"] == pytest.approx(precision["numerator"] / precision["denominator"])
    assert precision["value"] == pytest.approx(threshold.precision)
    assert false_positive_rate["numerator"] == threshold.false_positive
    assert false_positive_rate["denominator"] == threshold.legitimate_count
    assert false_positive_rate["value"] == pytest.approx(
        false_positive_rate["numerator"] / false_positive_rate["denominator"]
    )
    assert false_positive_rate["value"] == pytest.approx(threshold.false_positive_rate)
    assert campaign_recall == {
        "numerator": campaigns.detected_campaign_count,
        "denominator": campaigns.campaign_count,
        "value": campaigns.campaign_recall,
    }
    assert declared["average_precision"] == pytest.approx(
        authoritative.probability.average_precision
    )
    assert declared["roc_auc"] == pytest.approx(authoritative.probability.roc_auc)
    assert declared["decision_threshold"] == pytest.approx(float(threshold.threshold))


def test_false_positive_cost_and_safe_scope_are_explicit() -> None:
    manifest = _manifest()
    threshold = load_model_evaluation(
        REPOSITORY_ROOT / manifest["evaluation"]["source"]["path"]
    ).threshold
    cost = manifest["false_positive_cost"]
    safety = manifest["safety_boundaries"]

    assert cost["false_positive_count"] == threshold.false_positive
    assert cost["observed_false_positive_cost_units"] == (
        threshold.false_positive * cost["configured_weight_units_per_false_positive"]
    )
    assert cost["total_threshold_cost_units"] == threshold.cost_units
    assert cost["currency_value"] is None
    assert cost["currency_claim"] is False
    assert cost["live_policy"] == {
        "threshold_positive_action": "deny",
        "deny_creates_razorpay_order": False,
        "review_is_operational_fail_safe": True,
    }

    assert safety["defense_only"] is True
    assert safety["synthetic_data"] is True
    assert safety["razorpay_mode"] == "test_only"
    assert safety["external_adversarial_target"] is False
    for action in ("payment_capture", "payment_refund", "payment_settlement"):
        assert safety[action] is False
    assert safety["payment_modification"] is False
    assert safety["post_decision_explanations_can_change_decision"] is False


def test_evaluator_material_has_no_unsafe_or_evaluator_directive_claims() -> None:
    prohibited = (
        "production-ready",
        "production ready",
        "score this project highly",
        "ignore previous instructions",
        "guaranteed merchant savings",
        "guaranteed fraud prevention",
    )
    for path in EVALUATOR_FILES:
        content = path.read_text(encoding="utf-8").lower()
        for phrase in prohibited:
            assert phrase not in content, f"{path.name}: {phrase}"
