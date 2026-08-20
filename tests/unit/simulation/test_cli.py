import json
import sys
from pathlib import Path

from riskloom.simulation.cli import main
from riskloom.simulation.config import GeneratorConfig
from riskloom.simulation.validation import validate_dataset_directory


def test_generate_command_publishes_valid_artifacts(
    tiny_config: GeneratorConfig,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(tiny_config.model_dump(mode="json")),
        encoding="utf-8",
    )
    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "riskloom.simulation",
            "generate",
            "--config",
            str(config_path),
            "--seed",
            "20260820",
            "--output-dir",
            str(output),
        ],
    )

    main()

    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "generated"
    assert response["event_count"] == 300
    assert validate_dataset_directory(output)["status"] == "valid"


def test_replay_command_consumes_events_only(
    tiny_output: tuple[Path, object],
    monkeypatch,
    capsys,
) -> None:
    output, _ = tiny_output
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "riskloom.simulation",
            "replay",
            "--events",
            str(output / "events.jsonl"),
            "--timing",
            "no-delay",
            "--maximum-events",
            "2",
        ],
    )

    main()

    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "replayed"
    assert response["events_emitted"] == 2
    assert response["first_occurred_at"] <= response["last_occurred_at"]
