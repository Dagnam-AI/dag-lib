"""The tested SDK ranges: which version is in range, and how a version is read."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Optional

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


def test_the_declared_ranges_match_the_dev_dependencies() -> None:
    assert _compat.OPENAI == (">=1.66", "<3")
    assert _compat.ANTHROPIC == (">=0.40", "<1")


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
