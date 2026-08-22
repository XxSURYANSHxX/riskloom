"""No Gate C1 policy band is reachable from a live decision, and no PII can be submitted."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from riskloom.serving.schemas import CheckoutPreflightRequest
from riskloom.simulation.event_schema import Channel

REPOSITORY_ROOT = Path(__file__).parents[3]
SERVING_SOURCE = REPOSITORY_ROOT / "src/riskloom/serving"
LIVE_SOURCES = (
    SERVING_SOURCE,
    REPOSITORY_ROOT / "src/riskloom/api",
    REPOSITORY_ROOT / "src/riskloom/api/routes",
)


def _valid_request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_id": "evt_" + "1" * 32,
        "merchant_id": "mrc_" + "2" * 32,
        "checkout_id": "chk_" + "3" * 32,
        "customer_token": None,
        "device_token": "dev_" + "4" * 32,
        "network_token": "net_" + "5" * 32,
        "session_token": "ses_" + "6" * 32,
        "payment_instrument_token": "pmt_" + "7" * 32,
        "amount_subunits": 25_000,
        "currency": "INR",
        "channel": "web",
    }
    payload.update(overrides)
    return payload


def test_no_live_source_imports_a_policy_band_or_the_training_module() -> None:
    forbidden = ("riskloom.policy", "riskloom.modeling.training")
    for directory in LIVE_SOURCES:
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported.update(
                node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            )
            assert not any(
                name == blocked or name.startswith(f"{blocked}.")
                for name in imported
                for blocked in forbidden
            ), path
            # Identifiers, not raw text: prose explaining *why* a module must not import a
            # policy band is legitimate documentation and must not trip this check.
            referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            }
            assert not referenced.intersection(
                {"bands", "select_band", "BandPolicy", "CostPolicy", "evaluate_band"}
            ), path


def test_importing_the_live_decision_path_never_loads_a_policy_band() -> None:
    """Transitive proof in a fresh interpreter.

    ``riskloom.modeling.training`` does import ``riskloom.policy.bands`` for its Gate C1 boundary
    diagnostic, so an accidental import of that module would quietly make a policy band reachable
    from a live decision. This asserts neither ever loads.
    """

    probe = (
        "import sys\n"
        "import riskloom.serving.decisions\n"
        "import riskloom.serving.engine_host\n"
        "import riskloom.serving.model_host\n"
        "import riskloom.serving.schemas\n"
        "import riskloom.services.preflight\n"
        "import riskloom.api.routes.checkout\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('riskloom.policy')\n"
        "    or name == 'riskloom.modeling.training'\n"
        ")\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    assert completed.stdout.strip() == "LEAKED="


def test_only_one_risk_threshold_exists_in_the_decision_module() -> None:
    """A second risk threshold would be a policy band by another name."""

    source = (SERVING_SOURCE / "decisions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    threshold_comparisons = [
        node
        for node in comparisons
        if any(
            isinstance(operand, ast.Name) and operand.id == "decision_threshold"
            for operand in [*node.comparators, node.left]
        )
    ]
    assert len(threshold_comparisons) == 1


@pytest.mark.parametrize(
    "field",
    ["email", "contact", "card", "cvv", "ip_address", "vpa", "name", "notes", "description"],
)
def test_request_schema_rejects_personally_identifying_fields(field: str) -> None:
    with pytest.raises(ValueError):
        CheckoutPreflightRequest.model_validate(_valid_request(**{field: "anything"}))


def test_request_schema_rejects_a_client_supplied_timestamp_or_outcome() -> None:
    for field, value in (
        ("occurred_at", "2026-03-01T00:00:00Z"),
        ("outcome", "authorized"),
        ("failure_category", None),
    ):
        with pytest.raises(ValueError):
            CheckoutPreflightRequest.model_validate(_valid_request(**{field: value}))


def test_request_schema_rejects_malformed_tokens() -> None:
    for field, value in (
        ("device_token", "person@example.invalid"),
        ("merchant_id", "mrc_not_hex"),
        ("session_token", "192.0.2.1"),
    ):
        with pytest.raises(ValueError):
            CheckoutPreflightRequest.model_validate(_valid_request(**{field: value}))


def test_request_schema_rejects_a_cross_type_token_swap() -> None:
    """A well-formed token of the wrong type must not satisfy another field.

    The three values in the malformed-token test above are grossly malformed and violate several
    parts of the pattern at once, so none of them isolates the prefix. This one does: the value is
    a perfectly valid device token, correct length and all-hex, placed in ``merchant_id``. Only the
    prefix is wrong, so a rejection here can only be the prefix check.
    """

    swapped = "dev_" + "4" * 32
    with pytest.raises(ValidationError) as error:
        CheckoutPreflightRequest.model_validate(_valid_request(merchant_id=swapped))

    entries = error.value.errors()
    assert len(entries) == 1
    assert entries[0]["type"] == "string_pattern_mismatch"
    assert entries[0]["loc"] == ("merchant_id",)

    # Control: the very same value is accepted in the field it belongs to, so the rejection above
    # is about placement and cannot be blamed on the value being malformed.
    accepted = CheckoutPreflightRequest.model_validate(_valid_request(device_token=swapped))
    assert accepted.device_token == swapped


def test_response_never_carries_features_or_internal_state() -> None:
    from riskloom.serving.schemas import PreflightDecisionResponse  # noqa: PLC0415

    fields = set(PreflightDecisionResponse.model_fields)
    assert fields == {
        "decision_id",
        "event_id",
        "action",
        "risk_decision",
        "calibrated_probability",
        "decision_threshold",
        "model_id",
        "fail_safe_reason",
        "razorpay_order_id",
        "evaluated_at",
        "duplicate",
    }
    for leaked in ("features", "feature_vector", "diagnostics", "state", "merchant_id"):
        assert leaked not in fields


def test_amount_subunits_has_no_local_minimum_beyond_positivity() -> None:
    """Addition 3: a sub-minimum amount must reach Razorpay unmodified.

    Razorpay documents an INR minimum of 100 paise. The service deliberately does not enforce that
    locally, so the manual verification's REVIEW step is a genuine upstream rejection rather than a
    locally fabricated one.
    """

    request = CheckoutPreflightRequest.model_validate(_valid_request(amount_subunits=1))
    assert request.amount_subunits == 1

    with pytest.raises(ValueError):
        CheckoutPreflightRequest.model_validate(_valid_request(amount_subunits=0))

    # And nothing between the request and the client rewrites it.
    preflight_source = (REPOSITORY_ROOT / "src/riskloom/services/preflight.py").read_text(
        encoding="utf-8"
    )
    assert "amount=request.amount_subunits" in preflight_source


def test_channel_is_restricted_to_the_trained_enumeration() -> None:
    assert {member.value for member in Channel} == {"web", "mobile_web", "mobile_app"}
    with pytest.raises(ValueError):
        CheckoutPreflightRequest.model_validate(_valid_request(channel="pos"))
