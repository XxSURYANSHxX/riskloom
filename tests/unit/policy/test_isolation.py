import ast
import inspect
import subprocess
import sys
from pathlib import Path

import riskloom.policy
from riskloom.policy import bands

POLICY_SOURCE = Path(__file__).parents[3] / "src/riskloom/policy"


def test_policy_package_has_no_label_network_or_service_import_surface() -> None:
    forbidden_modules = {
        "asyncpg",
        "fastapi",
        "httpx",
        "importlib",
        "joblib",
        "pickle",
        "requests",
        "socket",
        "sqlalchemy",
        "subprocess",
        "urllib",
        "riskloom.api",
        "riskloom.db",
        "riskloom.integrations",
        "riskloom.services",
        "riskloom.simulation",
    }
    for path in POLICY_SOURCE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        assert not any(
            imported == forbidden or imported.startswith(f"{forbidden}.")
            for imported in imports
            for forbidden in forbidden_modules
        ), path
        lowered = source.casefold()
        assert "label_schema" not in lowered, path
        assert "razorpay" not in lowered, path
        assert "eval(" not in lowered, path
        assert "__import__" not in lowered, path


def test_importing_policy_does_not_load_label_schema() -> None:
    """Transitive proof, not just a per-file one: nothing the package imports pulls labels in.

    This runs in a fresh interpreter rather than mutating ``sys.modules`` in-process. Clearing the
    module table here would hand later tests stale module objects and break their monkeypatching,
    and a subprocess is a stronger proof anyway: nothing this session already imported can mask a
    missing import.
    """

    probe = (
        "import sys\n"
        "import riskloom.policy\n"
        "import riskloom.policy.artifacts\n"
        "import riskloom.policy.bands\n"
        "import riskloom.policy.comparison\n"
        "import riskloom.policy.canonical\n"
        "import riskloom.policy.config\n"
        "leaked = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('riskloom.simulation') or name.startswith('riskloom.db')\n"
        "    or name.startswith('riskloom.api') or name.startswith('riskloom.integrations')\n"
        ")\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parents[3],
    )
    assert completed.stdout.strip() == "LEAKED="


def test_routing_surface_accepts_only_probabilities_and_a_band() -> None:
    """The inference-time decision function cannot be handed an answer key."""

    parameters = list(inspect.signature(bands.band_decisions).parameters)
    assert parameters == ["probabilities", "band"]
    band_fields = set(bands.BandPolicy.model_fields)
    assert band_fields == {"lower_threshold", "upper_threshold"}


def test_policy_source_never_names_a_label_derived_field() -> None:
    forbidden_names = (
        "scenario_type",
        "campaign_id",
        "is_attack",
        "generator_metadata",
        "scenario_instance_id",
        "split",
    )
    for path in POLICY_SOURCE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        names.update(
            argument.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            for argument in node.args.args
        )
        assert not names.intersection(forbidden_names), path


def test_policy_public_surface_is_stable() -> None:
    assert set(riskloom.policy.__all__) == {
        "ALLOW",
        "DENY",
        "REVIEW",
        "BandPolicy",
        "CostPolicy",
        "PolicyConfig",
        "band_decisions",
        "evaluate_band",
        "evaluate_single_threshold",
        "load_policy_config",
        "select_band",
    }
