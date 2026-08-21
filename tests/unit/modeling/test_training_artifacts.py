import json
from pathlib import Path

import pytest

from riskloom.modeling.artifacts import (
    ModelingArtifactError,
    load_locked_model,
    publish_evaluation,
    publish_model,
    require_output_separate,
    runtime_dependency_versions,
)
from riskloom.modeling.canonical import canonical_json_bytes
from riskloom.modeling.config import ModelingConfig
from riskloom.modeling.data import EvaluationData, TrainingData, ValidationData
from riskloom.modeling.evaluation import build_evaluation_report, evaluate_model
from riskloom.modeling.training import TrainingOutputs, _ranking_key, train_model, validate_model


def test_training_report_contains_only_pre_lock_partitions(
    training_outputs: TrainingOutputs,
) -> None:
    report = training_outputs.report
    assert set(report["partitions"]) == {"train", "calibration_fit", "policy_selection"}
    assert report["selected_candidate"] in {"logistic_regression", "gradient_boosting"}
    assert "predictions" not in str(report)
    assert "event_id" not in str(report)

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for nested in value.values() for key in keys(nested)}
        if isinstance(value, list):
            return {key for nested in value for key in keys(nested)}
        return set()

    report_keys = keys(report)
    assert "test" not in report_keys
    assert not any(key.startswith("test_") for key in report_keys)
    assert "campaign_id" not in report_keys


def test_candidate_tie_break_order_is_complete_and_stable() -> None:
    baseline = {
        "cost_units": 10,
        "false_positive": 2,
        "precision": 0.5,
        "true_positive": 4,
    }
    assert _ranking_key(("a", 0.5, baseline)) < _ranking_key(
        ("b", 0.5, {**baseline, "cost_units": 11})
    )
    assert _ranking_key(("a", 0.5, baseline)) < _ranking_key(
        ("b", 0.5, {**baseline, "false_positive": 3})
    )
    assert _ranking_key(("a", 0.5, baseline)) < _ranking_key(
        ("b", 0.5, {**baseline, "true_positive": 3})
    )
    assert _ranking_key(("a", 0.5, baseline)) < _ranking_key(
        ("b", 0.5, {**baseline, "precision": 0.4})
    )
    assert _ranking_key(("a", 0.6, baseline)) < _ranking_key(("b", 0.5, baseline))
    assert _ranking_key(("a", 0.5, baseline)) < _ranking_key(("b", 0.5, baseline))


def test_candidate_ranking_treats_undefined_precision_as_zero() -> None:
    """`_ranking_key` is a separate ladder from `select_threshold`; pin its None fallback."""

    undefined = {"cost_units": 10, "false_positive": 0, "precision": None, "true_positive": 0}
    assert _ranking_key(("a", 0.5, {**undefined, "precision": 0.5})) < _ranking_key(
        ("a", 0.5, undefined)
    )
    assert _ranking_key(("a", 0.5, undefined)) == _ranking_key(
        ("a", 0.5, {**undefined, "precision": 0.0})
    )
    assert _ranking_key(("a", 0.6, undefined)) < _ranking_key(("b", 0.5, undefined))


def test_model_publication_is_canonical_manifest_last_and_no_overwrite(
    tmp_path: Path,
    modeling_config: ModelingConfig,
    training_outputs: TrainingOutputs,
) -> None:
    output = tmp_path / "model"
    result = publish_model(
        output,
        training_outputs.model,
        training_outputs.report,
        modeling_config,
        runtime_dependency_versions(),
    )
    assert result.artifact_id == training_outputs.model.model_id
    assert set(path.name for path in output.iterdir()) == {
        "model.json",
        "training_report.json",
        "manifest.json",
    }
    loaded, report, _ = load_locked_model(output, modeling_config)
    assert loaded == training_outputs.model
    assert report == training_outputs.report
    with pytest.raises(ModelingArtifactError, match="not_empty"):
        publish_model(
            output,
            training_outputs.model,
            training_outputs.report,
            modeling_config,
            runtime_dependency_versions(),
        )


def test_locked_model_rejects_tampering(
    tmp_path: Path,
    modeling_config: ModelingConfig,
    training_outputs: TrainingOutputs,
) -> None:
    output = tmp_path / "model"
    publish_model(
        output,
        training_outputs.model,
        training_outputs.report,
        modeling_config,
        runtime_dependency_versions(),
    )
    (output / "model.json").write_bytes((output / "model.json").read_bytes() + b" ")
    with pytest.raises(ModelingArtifactError):
        load_locked_model(output, modeling_config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model_schema", "schema_invalid"),
        ("manifest_marker", "marker_invalid"),
        ("configuration", "configuration_mismatch"),
        ("configuration_hash", "configuration_hash_mismatch"),
        ("artifacts_type", "manifest_artifacts_invalid"),
        ("artifact_entry", "manifest_artifacts_invalid"),
        ("artifact_hash", "artifact_hash_mismatch"),
        ("identity", "identity_mismatch"),
    ],
)
def test_locked_model_fails_closed_for_canonical_tampering(
    tmp_path: Path,
    modeling_config: ModelingConfig,
    training_outputs: TrainingOutputs,
    mutation: str,
    message: str,
) -> None:
    output = tmp_path / mutation
    publish_model(
        output,
        training_outputs.model,
        training_outputs.report,
        modeling_config,
        runtime_dependency_versions(),
    )
    model = json.loads((output / "model.json").read_bytes())
    manifest = json.loads((output / "manifest.json").read_bytes())
    if mutation == "model_schema":
        model["unknown"] = 1
        (output / "model.json").write_bytes(canonical_json_bytes(model))
    elif mutation == "manifest_marker":
        manifest["product"] = "NotRiskLoom"
    elif mutation == "configuration":
        manifest["effective_modeling_configuration"]["false_negative_cost_units"] = 24
    elif mutation == "configuration_hash":
        manifest["effective_modeling_configuration_sha256"] = "0" * 64
    elif mutation == "artifacts_type":
        manifest["artifacts"] = []
    elif mutation == "artifact_entry":
        manifest["artifacts"]["model.json"] = None
    elif mutation == "artifact_hash":
        manifest["artifacts"]["model.json"]["sha256"] = "0" * 64
    else:
        manifest["model_id"] = "0" * 64
    if mutation != "model_schema":
        (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ModelingArtifactError, match=message):
        load_locked_model(output, modeling_config)


def test_locked_model_requires_exact_artifact_set(
    tmp_path: Path, modeling_config: ModelingConfig
) -> None:
    with pytest.raises(ModelingArtifactError, match="directory_unreadable"):
        load_locked_model(tmp_path / "missing", modeling_config)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "unknown").write_text("x", encoding="utf-8")
    with pytest.raises(ModelingArtifactError, match="artifact_set"):
        load_locked_model(incomplete, modeling_config)


def test_fit_free_evaluation_and_publication_use_only_locked_model(
    tmp_path: Path,
    modeling_config: ModelingConfig,
    training_outputs: TrainingOutputs,
    tiny_training_data: TrainingData,
) -> None:
    partition = tiny_training_data.policy_selection
    data = EvaluationData(
        partition.features,
        partition.targets,
        partition.scenarios,
        partition.campaign_ids,
        partition.occurred_at_ms,
    )
    before = training_outputs.model.model_dump_json()
    report = build_evaluation_report(
        training_outputs.model,
        data,
        false_positive_cost=1,
        false_negative_cost=25,
        reliability_bin_count=10,
    )
    assert training_outputs.model.model_dump_json() == before
    result = publish_evaluation(
        tmp_path / "evaluation",
        report,
        training_outputs.model,
        modeling_config,
        "d" * 64,
        "e" * 64,
    )
    assert len(result.artifact_id) == 64
    assert set(path.name for path in result.output_directory.iterdir()) == {
        "evaluation.json",
        "manifest.json",
    }


def test_modeling_output_refuses_files_and_repository_root(
    tmp_path: Path,
    modeling_config: ModelingConfig,
    training_outputs: TrainingOutputs,
) -> None:
    target = tmp_path / "target"
    target.write_text("unknown", encoding="utf-8")
    with pytest.raises(ModelingArtifactError, match="not_directory"):
        publish_model(
            target,
            training_outputs.model,
            training_outputs.report,
            modeling_config,
            runtime_dependency_versions(),
        )
    with pytest.raises(ModelingArtifactError, match="unsafe"):
        publish_model(
            Path.cwd(),
            training_outputs.model,
            training_outputs.report,
            modeling_config,
            runtime_dependency_versions(),
        )


def test_modeling_output_refuses_source_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ModelingArtifactError, match="path_overlap"):
        require_output_separate(source / "nested", (source,))
    with pytest.raises(ModelingArtifactError, match="path_overlap"):
        require_output_separate(tmp_path, (source,))


def test_public_training_workflow_publishes_only_after_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    tiny_training_data: TrainingData,
    training_outputs: TrainingOutputs,
) -> None:
    monkeypatch.setattr(
        "riskloom.modeling.training.load_modeling_config", lambda _: modeling_config
    )
    monkeypatch.setattr(
        "riskloom.modeling.training.load_training_data",
        lambda *_args, **_kwargs: tiny_training_data,
    )
    monkeypatch.setattr(
        "riskloom.modeling.training.build_training_outputs", lambda *_: training_outputs
    )
    result = train_model(Path("sim"), Path("features"), Path("config"), tmp_path / "model")
    assert result.artifact_id == training_outputs.model.model_id


def test_independent_validation_retrains_and_checks_held_out_feature_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    tiny_training_data: TrainingData,
    training_outputs: TrainingOutputs,
) -> None:
    model_directory = tmp_path / "model"
    publish_model(
        model_directory,
        training_outputs.model,
        training_outputs.report,
        modeling_config,
        runtime_dependency_versions(),
    )
    validation_data = ValidationData(
        training=tiny_training_data,
        held_out_feature_sample=tiny_training_data.policy_selection.features[:3],
    )
    monkeypatch.setattr(
        "riskloom.modeling.training.load_modeling_config", lambda _: modeling_config
    )
    monkeypatch.setattr(
        "riskloom.modeling.training.load_training_data", lambda *_args, **_kwargs: validation_data
    )
    monkeypatch.setattr(
        "riskloom.modeling.training.build_training_outputs", lambda *_: training_outputs
    )
    result = validate_model(Path("sim"), Path("features"), Path("config"), model_directory)
    assert result["status"] == "valid"
    assert result["parity"]["held_out_features"]["checked_row_count"] == 3


def test_evaluate_workflow_is_fit_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modeling_config: ModelingConfig,
    tiny_training_data: TrainingData,
    training_outputs: TrainingOutputs,
) -> None:
    partition = tiny_training_data.policy_selection
    evaluation_data = EvaluationData(
        partition.features,
        partition.targets,
        partition.scenarios,
        partition.campaign_ids,
        partition.occurred_at_ms,
    )
    monkeypatch.setattr(
        "riskloom.modeling.evaluation.load_modeling_config", lambda _: modeling_config
    )
    monkeypatch.setattr(
        "riskloom.modeling.evaluation.load_locked_model",
        lambda *_: (
            training_outputs.model,
            training_outputs.report,
            {"artifacts": {"model.json": {"sha256": "d" * 64}}},
        ),
    )
    monkeypatch.setattr(
        "riskloom.modeling.evaluation.load_evaluation_data", lambda *_: evaluation_data
    )
    monkeypatch.setattr(
        "riskloom.modeling.evaluation.file_metadata", lambda _: {"sha256": "e" * 64}
    )
    result = evaluate_model(
        Path("sim"),
        Path("features"),
        Path("config"),
        Path("model"),
        tmp_path / "evaluation",
    )
    assert result.output_directory.name == "evaluation"
