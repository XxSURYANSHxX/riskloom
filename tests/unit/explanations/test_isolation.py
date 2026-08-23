"""The explanation path and the decision path must not be able to reach each other.

Enforced in both directions and at two levels: a static AST check over the sources, and a
transitive check in a fresh interpreter. The second matters because an import chain several modules
deep would satisfy the first while still linking the two paths at runtime.
"""

import ast
import subprocess
import sys
from pathlib import Path

from riskloom.explanations.schemas import FailSafeReasonInput
from riskloom.serving.schemas import FailSafeReason

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY_ROOT / "src" / "riskloom"
GENERATION_PACKAGE = SOURCE / "explanations"
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


def test_the_generation_package_cannot_reach_the_decision_path() -> None:
    forbidden = (
        "riskloom.serving",
        "riskloom.services",
        "riskloom.policy",
        "riskloom.modeling",
        "riskloom.features",
        "riskloom.api",
    )
    for path in _sources(GENERATION_PACKAGE):
        assert not _blocked(_imports(path), forbidden), path


def test_the_generation_package_cannot_reach_a_database() -> None:
    """Write isolation by construction rather than by restraint.

    Nothing here holds a session or an ORM class, so this package *cannot* write to
    ``risk_decisions``. The capability is absent, not merely unused.
    """

    forbidden = ("sqlalchemy", "riskloom.db", "asyncpg", "alembic")
    for path in _sources(GENERATION_PACKAGE):
        assert not _blocked(_imports(path), forbidden), path
        # Identifiers, not raw text: a docstring explaining why the ledger is untouchable is
        # legitimate documentation and must not trip this check.
        assert not _identifiers(path).intersection(
            {"RiskDecision", "ReviewItem", "AsyncSession", "session", "execute", "commit"}
        ), path


def test_the_decision_path_cannot_reach_the_generation_package() -> None:
    for directory in DECISION_SOURCES:
        for path in _sources(directory):
            assert not _blocked(_imports(path), ("riskloom.explanations",)), path
            assert not _identifiers(path).intersection(
                {"GeminiClient", "ExplanationInput", "LlmExplanation", "explain"}
            ), path


def test_importing_the_decision_path_never_loads_the_generation_package() -> None:
    """Transitive proof in a fresh interpreter."""

    probe = (
        "import sys\n"
        "import riskloom.serving.decisions\n"
        "import riskloom.services.preflight\n"
        "import riskloom.api.routes.checkout\n"
        "leaked = sorted(n for n in sys.modules if n.startswith('riskloom.explanations'))\n"
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


def test_importing_the_generation_package_never_loads_the_decision_path() -> None:
    """The converse, including a proof that no ORM model is pulled in.

    Note the assertion is on ``riskloom.db``, not on ``sqlalchemy`` itself: ``riskloom.core.config``
    uses ``make_url`` to validate the database URL, so the library loads transitively through
    settings. That is not a ledger-write capability. The static test above proves no source in this
    package imports SQLAlchemy directly, and this one proves no ORM model is reachable.
    """

    probe = (
        "import sys\n"
        "import riskloom.explanations.client\n"
        "import riskloom.explanations.factors\n"
        "import riskloom.explanations.prompt\n"
        "import riskloom.explanations.sanitizer\n"
        "import riskloom.explanations.schemas\n"
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


def test_the_duplicated_fail_safe_enum_matches_the_serving_definition() -> None:
    """``FailSafeReasonInput`` is duplicated rather than imported, to keep the packages apart.

    Duplication is only safe while the two stay identical, so that is asserted rather than trusted.
    """

    assert {item.value for item in FailSafeReasonInput} == {item.value for item in FailSafeReason}


def test_no_generation_source_names_a_policy_band_or_a_threshold() -> None:
    """The explanation must not imply a second risk threshold exists."""

    for path in _sources(GENERATION_PACKAGE):
        assert not _identifiers(path).intersection(
            {"select_band", "BandPolicy", "CostPolicy", "decision_threshold_for", "decide"}
        ), path
