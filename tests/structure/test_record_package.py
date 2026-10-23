"""``record/``: the stdlib-only ``dagnam-record`` package is wired into every repo-wide gate.

The recorder runs on a customer's machine next to their training code. Its promise
is that it has no dependencies and never opens a network connection, so the
promise is checked here by reading its source (an AST walk), not by trust.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "record"
PACKAGE = RECORD / "dagnam_record"

# Anything that can open a connection. Banned in every scope, lazy imports included.
NETWORK_MODULES = frozenset({"requests", "urllib", "http", "socket", "ssl", "ftplib", "smtplib"})
# The SDK and its contract: banned at module level, so ``import dagnam_record`` never pulls
# them in. A lazy (function-level) import is allowed only where listed here, with its reason.
SDK_MODULES = frozenset({"dagnam", "dagnam_contracts"})
LAZY_ALLOWLIST: dict[tuple[str, str], str] = {
    ("dagnam_record/redact.py", "dagnam_contracts"): (
        "the redactor may load the optional PII class list lazily, only when it is installed"
    ),
}


def _toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _imported_roots(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _walk(node: ast.AST, lazy: bool) -> list[tuple[str, bool]]:
    """Every ``(root module, is_lazy)`` imported under ``node``; lazy means inside a function."""
    found = [(root, lazy) for root in _imported_roots(node)]
    for child in ast.iter_child_nodes(node):
        inside = lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        found.extend(_walk(child, inside))
    return found


def violations(source: str, relative: str) -> list[str]:
    """What in ``source`` (a module at ``relative`` in the package) breaks the import rules."""
    found: list[str] = []
    for root, lazy in _walk(ast.parse(source), False):
        if root in NETWORK_MODULES:
            found.append(f"{relative}: imports network module {root!r}")
        elif root in SDK_MODULES and not lazy:
            found.append(f"{relative}: imports {root!r} at module level")
        elif root in SDK_MODULES and (relative, root) not in LAZY_ALLOWLIST:
            found.append(f"{relative}: lazily imports {root!r} without an allowlist entry")
    return found


def test_the_package_metadata_promises_no_dependencies_and_python_39() -> None:
    project = _toml(RECORD / "pyproject.toml")["project"]
    assert project["name"] == "dagnam-record"
    assert project["dependencies"] == []
    assert project["requires-python"] == ">=3.9"
    assert project["license"] == "Apache-2.0"


def test_the_package_ships_typed_with_its_licence_files() -> None:
    assert (PACKAGE / "py.typed").is_file()
    assert (RECORD / "README.md").is_file()
    assert (RECORD / "CHANGELOG.md").is_file()
    assert (RECORD / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert (RECORD / "NOTICE").read_bytes() == (ROOT / "NOTICE").read_bytes()


def test_no_module_imports_the_network_or_the_sdk() -> None:
    modules = sorted(PACKAGE.rglob("*.py"))
    assert modules, "record/dagnam_record has no modules to check"
    found: list[str] = []
    for module in modules:
        relative = module.relative_to(RECORD).as_posix()
        found.extend(violations(module.read_text(encoding="utf-8"), relative))
    assert found == []


def test_the_import_walk_flags_each_kind_of_violation() -> None:
    assert violations("import socket\n", "a.py") == ["a.py: imports network module 'socket'"]
    assert violations("from urllib.request import urlopen\n", "a.py") == [
        "a.py: imports network module 'urllib'"
    ]
    assert violations("def f():\n    import requests\n", "a.py") == [
        "a.py: imports network module 'requests'"
    ]
    assert violations("import dagnam\n", "a.py") == ["a.py: imports 'dagnam' at module level"]
    assert violations(
        "try:\n    import dagnam_contracts\nexcept ImportError:\n    pass\n", "a.py"
    ) == ["a.py: imports 'dagnam_contracts' at module level"]
    assert violations("class C:\n    import dagnam\n", "a.py") == [
        "a.py: imports 'dagnam' at module level"
    ]
    assert violations("async def f():\n    import dagnam\n", "a.py") == [
        "a.py: lazily imports 'dagnam' without an allowlist entry"
    ]


def test_the_import_walk_allows_listed_lazy_imports_and_plain_ones() -> None:
    lazy = "def load():\n    import dagnam_contracts\n"
    assert violations(lazy, "dagnam_record/redact.py") == []
    assert violations(lazy, "dagnam_record/other.py") == [
        "dagnam_record/other.py: lazily imports 'dagnam_contracts' without an allowlist entry"
    ]
    assert violations("import json\nfrom . import sibling\nfrom os import path\n", "a.py") == []


def test_pyright_checks_the_package_against_its_oldest_python() -> None:
    pyright = _toml(ROOT / "pyproject.toml")["tool"]["pyright"]
    assert "record" in pyright["include"]
    environments = {env["root"]: env for env in pyright["executionEnvironments"]}
    assert environments["record"]["pythonVersion"] == "3.9"


def test_ruff_pytest_and_coverage_see_the_package() -> None:
    tool = _toml(ROOT / "pyproject.toml")["tool"]
    assert "dagnam_record" in tool["ruff"]["src"]
    assert "dagnam_record" in tool["ruff"]["lint"]["isort"]["known-first-party"]
    assert "record/tests" in tool["pytest"]["ini_options"]["testpaths"]
    assert "dagnam_record" in tool["coverage"]["run"]["source"]
    test_task = tool["poe"]["tasks"]["test"]["cmd"]
    assert "--cov=dagnam_record" in test_task
    assert "record/tests" in test_task


def test_import_linter_forbids_the_package_from_importing_the_sdk() -> None:
    importlinter = _toml(ROOT / "pyproject.toml")["tool"]["importlinter"]
    assert "dagnam_record" in importlinter["root_packages"]
    forbidden = [c for c in importlinter["contracts"] if c["type"] == "forbidden"]
    assert len(forbidden) == 1
    assert forbidden[0]["source_modules"] == ["dagnam_record"]
    assert set(forbidden[0]["forbidden_modules"]) == SDK_MODULES


def test_dependabot_and_code_owners_cover_the_package() -> None:
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert 'directory: "/record"' in dependabot
    owners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "/record/pyproject.toml" in owners
