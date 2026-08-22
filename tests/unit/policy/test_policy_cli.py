import json
from pathlib import Path
from typing import Any

import pytest

from riskloom.modeling.cli import main
from riskloom.policy.artifacts import PolicyPublicationResult

FIT_ARGUMENTS = [
    "fit-policy-band",
    "--simulation-dir",
    "sim",
    "--feature-dir",
    "features",
    "--config",
    "config.json",
    "--model-dir",
    "model",
    "--policy-config",
    "policy.json",
    "--output-dir",
    "band",
]

VALIDATE_ARGUMENTS = [
    "validate-policy",
    "--band-dir",
    "band",
    "--validation-simulation-dir",
    "vsim",
    "--validation-feature-dir",
    "vfeat",
    "--config",
    "config.json",
    "--model-dir",
    "model",
    "--policy-config",
    "policy.json",
    "--output-dir",
    "comparison",
]


def _comparison(*, eligible: bool, failed: list[str], approve: bool) -> dict[str, Any]:
    return {
        "approval": {
            "approval_granted": approve and eligible,
            "approval_requested": approve,
            "refusal_reasons": [] if eligible else failed,
        },
        "gates": {
            "approval_eligible": eligible,
            "beats_incumbent_cost": True,
            "cost_delta_units": -22,
            "failed_gates": failed,
        },
    }


def test_fit_policy_band_cli_returns_a_safe_aggregate(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "riskloom.modeling.cli.fit_policy_band",
        lambda *_: PolicyPublicationResult("a" * 64, Path("band"), {}),
    )
    assert main(FIT_ARGUMENTS) == 0
    assert capsys.readouterr().out == '{"band_id":"' + "a" * 64 + '","status":"band_fitted"}\n'


def test_validate_policy_without_approval_reports_without_granting(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "riskloom.modeling.cli.validate_policy",
        lambda *_a, **_k: (
            PolicyPublicationResult("b" * 64, Path("comparison"), {}),
            _comparison(eligible=True, failed=[], approve=False),
        ),
    )
    assert main(VALIDATE_ARGUMENTS) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["approval_eligible"] is True
    assert payload["approval_granted"] is False
    assert payload["status"] == "validated"


def test_validate_policy_grants_approval_only_with_the_explicit_flag(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "riskloom.modeling.cli.validate_policy",
        lambda *_a, **kwargs: (
            PolicyPublicationResult("c" * 64, Path("comparison"), {}),
            _comparison(eligible=True, failed=[], approve=kwargs["approve"]),
        ),
    )
    assert main([*VALIDATE_ARGUMENTS, "--approve"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["approval_granted"] is True


def test_validate_policy_refuses_approval_when_a_gate_failed(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    failed = ["banded_policy_exceeds_false_positive_rate_ceiling"]
    monkeypatch.setattr(
        "riskloom.modeling.cli.validate_policy",
        lambda *_a, **kwargs: (
            PolicyPublicationResult("d" * 64, Path("comparison"), {}),
            _comparison(eligible=False, failed=failed, approve=kwargs["approve"]),
        ),
    )
    # Requesting approval the evidence does not support is a failure, not a silent no-op.
    assert main([*VALIDATE_ARGUMENTS, "--approve"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["approval_granted"] is False
    assert payload["failed_gates"] == failed


def test_validate_policy_without_approval_still_succeeds_when_a_gate_failed(
    monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """A losing policy is a publishable result, not a command failure."""

    monkeypatch.setattr(
        "riskloom.modeling.cli.validate_policy",
        lambda *_a, **kwargs: (
            PolicyPublicationResult("e" * 64, Path("comparison"), {}),
            _comparison(
                eligible=False,
                failed=["banded_policy_does_not_beat_incumbent_cost"],
                approve=kwargs["approve"],
            ),
        ),
    )
    assert main(VALIDATE_ARGUMENTS) == 0
    assert json.loads(capsys.readouterr().out)["approval_eligible"] is False


def test_policy_cli_accepts_no_remote_or_label_arguments() -> None:
    from riskloom.modeling.cli import _parser  # noqa: PLC0415

    parser = _parser()
    destinations: set[str] = set()
    for action in parser._subparsers._actions:  # noqa: SLF001
        choices = getattr(action, "choices", {})
        if not isinstance(choices, dict):
            continue
        for name in ("fit-policy-band", "validate-policy"):
            if name in choices:
                destinations.update(item.dest for item in choices[name]._actions)  # noqa: SLF001
    assert {"labels", "url", "target", "host", "scenario", "campaign"}.isdisjoint(destinations)
    assert "approve" in destinations


@pytest.mark.parametrize("command", ["fit-policy-band", "validate-policy"])
def test_policy_cli_requires_every_path_argument(command: str) -> None:
    with pytest.raises(SystemExit):
        main([command])
