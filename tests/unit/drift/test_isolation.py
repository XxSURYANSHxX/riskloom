"""Drift must not be able to reach a decision, and a decision must not be able to reach drift.

Enforced in both directions and at two levels, matching the Day 6 and Day 8 pattern: a static AST
check over the sources, and a transitive check in a fresh interpreter. The second matters because
an import chain several modules deep satisfies the first while still linking the two at runtime.
"""

import ast
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY_ROOT / "src" / "riskloom"
DRIFT_PACKAGE = SOURCE / "drift"
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


def test_the_drift_package_cannot_reach_the_decision_path() -> None:
    forbidden = (
        "riskloom.serving",
        "riskloom.services",
        "riskloom.policy",
        "riskloom.modeling",
        "riskloom.features",
        "riskloom.api",
    )
    for path in _sources(DRIFT_PACKAGE):
        assert not _blocked(_imports(path), forbidden), path
        # Identifiers, not raw text: prose explaining why drift must never decide is legitimate
        # documentation and must not trip this check.
        assert not _identifiers(path).intersection(
            {"decide", "fail_safe", "select_band", "BandPolicy", "CostPolicy", "ServingBundle"}
        ), path


def test_the_drift_package_cannot_reach_a_database() -> None:
    """Write isolation by construction: the capability is absent, not merely unused."""

    forbidden = ("sqlalchemy", "riskloom.db", "asyncpg", "alembic")
    for path in _sources(DRIFT_PACKAGE):
        assert not _blocked(_imports(path), forbidden), path
        assert not _identifiers(path).intersection(
            {"RiskDecision", "ReviewItem", "AsyncSession", "session", "execute", "commit"}
        ), path


def test_the_drift_package_never_writes() -> None:
    for path in _sources(DRIFT_PACKAGE):
        source = path.read_text(encoding="utf-8").casefold()
        for statement in ("insert into", "update ", "delete from", ".commit(", "session.add"):
            assert statement not in source, (path, statement)


def test_the_drift_package_never_reaches_the_protected_test_partition() -> None:
    """The reference comes from the published aggregate only.

    The held-out partition is never opened, re-scored, or re-derived; only the already-computed
    bin counts in ``evaluation.json`` are read.
    """

    for path in _sources(DRIFT_PACKAGE):
        assert not _blocked(_imports(path), ("riskloom.modeling.data",)), path
        assert not _identifiers(path).intersection(
            {"evaluate_test", "load_evaluation_data", "held_out", "test_partition"}
        ), path


def test_the_decision_path_cannot_reach_the_drift_package() -> None:
    for directory in DECISION_SOURCES:
        for path in _sources(directory):
            assert not _blocked(_imports(path), ("riskloom.drift",)), path
            assert not _identifiers(path).intersection(
                {"population_stability_index", "evaluate_drift", "DriftReport"}
            ), path


def test_importing_the_decision_path_never_loads_the_drift_package() -> None:
    probe = (
        "import sys\n"
        "import riskloom.serving.decisions\n"
        "import riskloom.services.preflight\n"
        "import riskloom.api.routes.checkout\n"
        "leaked = sorted(n for n in sys.modules if n.startswith('riskloom.drift'))\n"
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


def test_importing_the_drift_package_never_loads_the_decision_path() -> None:
    """The converse, including a proof that no ORM model is pulled in.

    Asserted on ``riskloom.db`` rather than on ``sqlalchemy`` itself: nothing in the drift package
    imports SQLAlchemy directly (the static test above proves that), but a settings import can pull
    the library in transitively, and that is not a ledger-write capability.
    """

    probe = (
        "import sys\n"
        "import riskloom.drift.psi\n"
        "import riskloom.drift.reference\n"
        "import riskloom.drift.schemas\n"
        "leaked = sorted(\n"
        "    n for n in sys.modules\n"
        "    if n.startswith('riskloom.serving')\n"
        "    or n.startswith('riskloom.services')\n"
        "    or n.startswith('riskloom.policy')\n"
        "    or n.startswith('riskloom.modeling')\n"
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


def test_no_drift_source_names_a_threshold_or_a_policy_band() -> None:
    """Drift must not imply a second decision surface exists."""

    for path in _sources(DRIFT_PACKAGE):
        assert not _identifiers(path).intersection(
            {"decision_threshold", "risk_decision", "threshold"}
        ), path
