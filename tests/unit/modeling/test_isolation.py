import ast
from pathlib import Path


def test_modeling_package_has_no_prohibited_runtime_surface() -> None:
    root = Path("src/riskloom/modeling")
    prohibited_imports = {
        "asyncpg",
        "fastapi",
        "httpx",
        "socket",
        "sqlalchemy",
        "subprocess",
        "riskloom.api",
        "riskloom.db",
        "riskloom.integrations",
        "riskloom.services",
        "riskloom.simulation.generation",
    }
    prohibited_calls = {"eval", "exec", "__import__"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                assert not any(
                    name == prohibited or name.startswith(f"{prohibited}.")
                    for name in imported
                    for prohibited in prohibited_imports
                )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not any(
                    node.module == prohibited or node.module.startswith(f"{prohibited}.")
                    for prohibited in prohibited_imports
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in prohibited_calls


def test_modeling_package_has_no_unsafe_serialization_or_network_literals() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in Path("src/riskloom/modeling").glob("*.py")
    )
    for prohibited in (
        "cloudpickle",
        "joblib",
        "pickle.dump",
        "pickle.load",
        "http://",
        "https://",
    ):
        assert prohibited not in combined
