from pathlib import Path

from riskloom.modeling.artifacts import PublicationResult
from riskloom.modeling.cli import main


def test_train_cli_returns_safe_aggregate(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "riskloom.modeling.cli.train_model",
        lambda *_: PublicationResult("a" * 64, Path("out"), {}),
    )
    result = main(
        [
            "train",
            "--simulation-dir",
            "sim",
            "--feature-dir",
            "features",
            "--config",
            "config.json",
            "--output-dir",
            "model",
        ]
    )
    assert result == 0
    assert capsys.readouterr().out == '{"model_id":"' + "a" * 64 + '","status":"trained"}\n'


def test_validate_and_evaluate_cli_paths(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "riskloom.modeling.cli.validate_model",
        lambda *_: {"model_id": "b" * 64, "status": "valid"},
    )
    validate_result = main(
        [
            "validate-model",
            "--simulation-dir",
            "sim",
            "--feature-dir",
            "features",
            "--config",
            "config.json",
            "--model-dir",
            "model",
        ]
    )
    assert validate_result == 0
    assert '"status":"valid"' in capsys.readouterr().out

    monkeypatch.setattr(
        "riskloom.modeling.cli.evaluate_model",
        lambda *_: PublicationResult("c" * 64, Path("out"), {}),
    )
    evaluate_result = main(
        [
            "evaluate-test",
            "--simulation-dir",
            "sim",
            "--feature-dir",
            "features",
            "--config",
            "config.json",
            "--model-dir",
            "model",
            "--output-dir",
            "evaluation",
        ]
    )
    assert evaluate_result == 0
    assert '"status":"evaluated"' in capsys.readouterr().out


def test_cli_returns_safe_error(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    def fail(*_):  # type: ignore[no-untyped-def]
        raise ValueError("safe_failure")

    monkeypatch.setattr("riskloom.modeling.cli.train_model", fail)
    result = main(
        [
            "train",
            "--simulation-dir",
            "sim",
            "--feature-dir",
            "features",
            "--config",
            "config.json",
            "--output-dir",
            "model",
        ]
    )
    assert result == 1
    assert capsys.readouterr().out == '{"error":"safe_failure"}\n'
