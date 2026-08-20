import json

import pytest
import structlog

from riskloom.core.logging import (
    REDACTED,
    configure_logging,
    redact_sensitive_fields,
    request_id_from_header,
)


def test_log_processor_recursively_redacts_sensitive_keys() -> None:
    event = redact_sensitive_fields(
        None,
        "info",
        {
            "event": "synthetic",
            "secret": "must-disappear",
            "nested": {
                "authorization": "must-disappear-too",
                "safe": "kept",
            },
        },
    )

    assert event["secret"] == REDACTED
    assert event["nested"]["authorization"] == REDACTED
    assert event["nested"]["safe"] == "kept"


def test_structured_logging_does_not_emit_sensitive_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    structlog.get_logger("test").info(
        "safe_event",
        payload={"email": "person@example.invalid"},
        safe="visible",
    )

    output = json.loads(capsys.readouterr().out)
    assert output["payload"] == REDACTED
    assert output["safe"] == "visible"
    assert "person@example.invalid" not in json.dumps(output)


def test_request_id_accepts_only_bounded_safe_values() -> None:
    assert request_id_from_header("safe-request_1") == "safe-request_1"
    assert request_id_from_header("unsafe request") != "unsafe request"
    assert request_id_from_header("x" * 65) != "x" * 65
    assert request_id_from_header(None)
