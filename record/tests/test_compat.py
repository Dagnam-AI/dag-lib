"""The tested SDK ranges: which version is in range, and how a version is read."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Optional, Tuple

import pytest

from dagnam_record import _compat


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.66.0", True),
        ("1.66", True),
        ("2.54.0", True),
        ("2.99.99", True),
        ("1.65.9", False),
        ("1.40.0", False),
        ("3.0.0", False),
        ("3.0.0rc1", False),
        ("0.28.1", False),
        ("2.1.0.dev3", True),
        ("garbage", False),
        ("", False),
        ("1", False),
    ],
)
def test_openai_range(version: str, expected: bool) -> None:
    assert _compat.in_range(version, _compat.OPENAI) is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [("0.40.0", True), ("0.125.0", True), ("0.39.9", False), ("1.0.0", False), ("x", False)],
)
def test_anthropic_range(version: str, expected: bool) -> None:
    assert _compat.in_range(version, _compat.ANTHROPIC) is expected


ROOT_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def dev_bounds(name: str) -> Tuple[str, str]:
    """The lower and upper bound the repo's dev dependency group asks for ``name``."""
    text = ROOT_PYPROJECT.read_text(encoding="utf-8")
    found = re.search(rf'^\s*"{name}(>=[0-9.]+),(<[0-9.]+)",?$', text, re.MULTILINE)
    assert found is not None, f"no bounded {name} in the dev dependencies"
    return found.group(1), found.group(2)


@pytest.mark.skipif(not ROOT_PYPROJECT.is_file(), reason="the repo's root pyproject is not here")
def test_the_ranges_are_the_ones_the_dev_dependencies_test_against() -> None:
    assert dev_bounds("openai") == _compat.OPENAI == (">=1.66", "<3")
    assert dev_bounds("anthropic") == _compat.ANTHROPIC == (">=0.40", "<1")


def test_a_version_with_thousands_of_digits_is_out_of_range_not_an_error() -> None:
    assert _compat.in_range("1." + "9" * 5000, _compat.OPENAI) is False
    assert _compat.in_range("9" * 5000 + ".1", _compat.OPENAI) is False


def test_a_version_longer_than_64_characters_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    fake("openai", "1.70.0" + "0" * 58, monkeypatch)  # 64 characters: kept
    assert _compat.version_of("openai") == "1.70.0" + "0" * 58
    fake("openai", "1.70.0" + "0" * 59, monkeypatch)  # 65: unknown
    assert _compat.version_of("openai") == _compat.UNKNOWN


def test_a_module_whose_attribute_lookup_raises_has_an_unknown_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("openai")

    def explode(name: str) -> object:
        raise RuntimeError("a lazy attribute that fails")

    vars(module)["__getattr__"] = explode
    monkeypatch.setitem(sys.modules, "openai", module)
    assert _compat.version_of("openai") == _compat.UNKNOWN


def test_whether_an_sdk_is_untested_follows_its_imported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake("openai", "1.70.0", monkeypatch)
    assert _compat.is_untested("openai") is False
    fake("openai", "0.28.1", monkeypatch)
    assert _compat.is_untested("openai") is True
    fake("openai", None, monkeypatch)
    assert _compat.is_untested("openai") is True  # no readable version: not vouched for
    monkeypatch.setitem(sys.modules, "openai", None)
    assert _compat.is_untested("openai") is False  # not imported: nothing of it is patched
    assert _compat.is_untested("something-else") is False  # a provider without a range


def fake(name: str, version: Optional[object], monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType(name)
    if version is not None:
        vars(module).update(__version__=version)
    monkeypatch.setitem(sys.modules, name, module)


def test_the_version_is_read_from_the_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake("openai", "1.70.0", monkeypatch)
    assert _compat.version_of("openai") == "1.70.0"


@pytest.mark.parametrize("version", [None, 3, ""])
def test_a_module_without_a_readable_version_is_unknown(
    version: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake("openai", version, monkeypatch)
    assert _compat.version_of("openai") == _compat.UNKNOWN


def test_a_module_that_is_not_imported_has_no_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)  # an import of it fails
    assert _compat.version_of("openai") is None
    monkeypatch.delitem(sys.modules, "openai")
    assert _compat.version_of("openai") is None


def test_an_unknown_version_is_out_of_range() -> None:
    assert _compat.in_range(_compat.UNKNOWN, _compat.OPENAI) is False
