"""Offline analysis must not be able to reach a decision, or be reached from one.

Same two-layer pattern as the serving, explanation and drift packages: a static AST check over the
sources, and a transitive check in a fresh interpreter.
"""

import ast
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY_ROOT / "src" / "riskloom"
ANALYSIS = SOURCE / "analysis"
DECISION_SOURCES = (SOURCE / "serving", SOURCE / "services" / "preflight.py")


def _sources(target: Path) -> list[Path]:
    return sorted(target.glob("*.py")) if target.is_dir() else [target]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    return names


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def _blocked(names: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return [
        name
        for name in names
        for blocked in forbidden
        if name == blocked or name.startswith(f"{blocked}.")
    ]


def test_analysis_cannot_reach_the_decision_path() -> None:
    forbidden = ("riskloom.serving", "riskloom.services", "riskloom.policy", "riskloom.api")
    for path in _sources(ANALYSIS):
        assert not _blocked(_imports(path), forbidden), path
        assert not _identifiers(path).intersection(
            {"decide", "fail_safe", "select_band", "BandPolicy", "evaluate_preflight"}
        ), path


def test_analysis_cannot_reach_a_database() -> None:
    forbidden = ("sqlalchemy", "riskloom.db", "asyncpg", "alembic")
    for path in _sources(ANALYSIS):
        assert not _blocked(_imports(path), forbidden), path


def test_analysis_never_trains_or_re_locks_a_model() -> None:
    """This gate scores a locked artifact. Any fitting call here would invalidate it."""

    for path in _sources(ANALYSIS):
        assert not _blocked(_imports(path), ("sklearn", "riskloom.modeling.training")), path
        assert not _identifiers(path).intersection(
            {"fit", "train_model", "publish_model", "select_threshold", "partial_fit"}
        ), path


def test_analysis_never_opens_the_held_out_evaluation_path() -> None:
    """Reading the published aggregate is fine; re-scoring the test partition is not."""

    for path in _sources(ANALYSIS):
        assert not _blocked(_imports(path), ("riskloom.modeling.data",)), path
        assert not _identifiers(path).intersection(
            {"evaluate_test", "load_evaluation_data", "load_held_out_features"}
        ), path


def test_the_decision_path_cannot_reach_analysis() -> None:
    for directory in DECISION_SOURCES:
        for path in _sources(directory):
            assert not _blocked(_imports(path), ("riskloom.analysis",)), path


def test_importing_the_decision_path_never_loads_analysis() -> None:
    probe = (
        "import sys\n"
        "import riskloom.serving.decisions\n"
        "import riskloom.services.preflight\n"
        "import riskloom.api.routes.checkout\n"
        "leaked = sorted(n for n in sys.modules if n.startswith('riskloom.analysis'))\n"
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


def test_importing_analysis_never_loads_the_decision_path() -> None:
    probe = (
        "import sys\n"
        "import riskloom.analysis.adversarial_stress\n"
        "import riskloom.analysis.references\n"
        "leaked = sorted(\n"
        "    n for n in sys.modules\n"
        "    if n.startswith('riskloom.serving')\n"
        "    or n.startswith('riskloom.services')\n"
        "    or n.startswith('riskloom.policy')\n"
        "    or n.startswith('riskloom.db')\n"
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
