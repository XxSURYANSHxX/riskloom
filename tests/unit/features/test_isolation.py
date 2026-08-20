import ast
import importlib
import sys
from pathlib import Path

from riskloom.features.cli import build_parser


def test_feature_package_has_no_label_network_or_service_import_surface() -> None:
    source_directory = Path(__file__).parents[3] / "src/riskloom/features"
    forbidden_modules = {
        "fastapi",
        "httpx",
        "importlib",
        "requests",
        "socket",
        "subprocess",
        "sqlalchemy",
        "urllib",
        "riskloom.db",
        "riskloom.integrations",
        "riskloom.simulation.artifacts",
        "riskloom.simulation.config",
        "riskloom.simulation.generation",
        "riskloom.simulation.label_schema",
        "riskloom.simulation.reporting",
        "riskloom.simulation.validation",
    }
    for path in source_directory.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
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
        source = path.read_text(encoding="utf-8").casefold()
        assert "label_schema" not in source
        assert "razorpay" not in source
        assert "eval(" not in source
        assert "__import__" not in source


def test_importing_features_does_not_load_label_schema() -> None:
    label_module = "riskloom.simulation.label_schema"
    sys.modules.pop(label_module, None)
    importlib.reload(importlib.import_module("riskloom.features"))
    importlib.reload(importlib.import_module("riskloom.features.cli"))
    assert label_module not in sys.modules


def test_outcome_is_read_only_for_update_and_prohibited_event_values_are_never_read() -> None:
    source = (Path(__file__).parents[3] / "src/riskloom/features/engine.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    assert attributes.count("outcome") == 1
    assert "failure_category" not in attributes
    assert "currency" not in attributes


def test_cli_accepts_no_labels_manifest_url_or_remote_target() -> None:
    parser = build_parser()
    destinations = {
        action.dest
        for action in parser._actions
        if action.dest != "help"  # noqa: SLF001
    }
    for subparser_action in parser._subparsers._actions:  # noqa: SLF001
        choices = getattr(subparser_action, "choices", {})
        if not isinstance(choices, dict):
            continue
        for subparser in choices.values():
            destinations.update(action.dest for action in subparser._actions)  # noqa: SLF001
    assert {"labels", "manifest", "url", "target", "host"}.isdisjoint(destinations)
