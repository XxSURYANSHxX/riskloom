import json
from pathlib import Path

from riskloom.features.cli import run
from riskloom.simulation.event_schema import CheckoutAttemptEvent


def test_extract_and_validate_cli_are_event_only(
    tmp_path: Path,
    tiny_events_path: Path,
    capsys,
) -> None:
    config_path = Path(__file__).parents[3] / "configs/features/default.json"
    output = tmp_path / "features"
    assert (
        run(
            [
                "extract",
                "--events",
                str(tiny_events_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    extracted = json.loads(capsys.readouterr().out)
    assert extracted["status"] == "generated"
    assert extracted["feature_count"] == 75

    assert (
        run(
            [
                "validate",
                "--events",
                str(tiny_events_path),
                "--config",
                str(config_path),
                "--input-dir",
                str(output),
            ]
        )
        == 0
    )
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "valid"


def test_cli_returns_safe_error(
    tmp_path: Path, tiny_events: list[CheckoutAttemptEvent], capsys
) -> None:
    del tiny_events
    assert (
        run(
            [
                "validate",
                "--events",
                str(tmp_path / "missing-events.jsonl"),
                "--config",
                str(tmp_path / "missing-config.json"),
                "--input-dir",
                str(tmp_path / "missing-features"),
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error == {"error": "feature_configuration_invalid", "status": "error"}
