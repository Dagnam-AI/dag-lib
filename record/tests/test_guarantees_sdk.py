"""The recorder against SDKs it was not tested with: other versions, partial installs, renames.

``openai>=1.66,<3`` and ``anthropic>=0.40,<1`` are the tested ranges (see ``_compat``). Outside
them the recorder writes the normalized form only and says so once; whatever else is wrong with
an SDK, ``install`` still returns and the program's calls still work.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from _guarantee_support import run_call
import anthropic
import openai
from openai.resources.chat.completions import Completions
import pytest
from typing_extensions import override

import dagnam_record
from dagnam_record import _compat, _openai
from dagnam_record._recorder import Recorder
from dagnam_record._writer import Writer

if TYPE_CHECKING:
    from _mock_server import MockServer

pytestmark = pytest.mark.usefixtures("fresh_state")


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real ``~/.dagnam``."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def lines_and_status(directory: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dagnam_record.uninstall()
    lines = [
        json.loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return lines, json.loads((directory / ".status.json").read_text(encoding="utf-8"))


def test_the_versions_in_use_are_saved_and_the_raw_bodies_are_kept_inside_the_range(
    tmp_path: Path, mock_server: MockServer
) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    run_call("openai ok", mock_server)
    run_call("anthropic ok", mock_server)
    lines, status = lines_and_status(tmp_path / "t")
    assert status["sdk_versions"] == {
        "openai": openai.__version__,
        "anthropic": anthropic.__version__,
    }
    assert [line["provider"] for line in lines] == ["openai", "anthropic"]
    assert all("request" in line and "response" in line and "norm" in line for line in lines)


@pytest.mark.parametrize("version", ["0.28.1", "1.65.0", "3.0.0", "unreadable"])
def test_an_openai_outside_the_range_is_recorded_as_norm_only_and_said_once(
    version: str,
    tmp_path: Path,
    mock_server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(openai, "__version__", version)
    with caplog.at_level(logging.WARNING, logger="dagnam_record"):
        dagnam_record.install(dir=str(tmp_path / "t"))
        dagnam_record.install(dir=str(tmp_path / "t"))  # again: still one warning
        run_call("openai ok", mock_server)
        run_call("anthropic ok", mock_server)
    assert caplog.text.count("outside the tested range") == 1
    assert version in caplog.text
    lines, status = lines_and_status(tmp_path / "t")
    by_provider = {line["provider"]: line for line in lines}
    assert "request" not in by_provider["openai"]
    assert "response" not in by_provider["openai"]
    assert by_provider["openai"]["norm"]["response"] == "Hello there"
    assert "request" in by_provider["anthropic"]  # the other SDK is inside its range
    assert status["sdk_versions"]["openai"] == version


def test_an_anthropic_outside_the_range_is_norm_only_and_openai_is_not(
    tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(anthropic, "__version__", "1.0.0")
    dagnam_record.install(dir=str(tmp_path / "t"))
    run_call("openai ok", mock_server)
    run_call("anthropic ok", mock_server)
    lines, _ = lines_and_status(tmp_path / "t")
    by_provider = {line["provider"]: line for line in lines}
    assert "request" in by_provider["openai"]
    assert "request" not in by_provider["anthropic"]
    assert by_provider["anthropic"]["norm"]["response"] == "Hello there"


def test_the_tested_range_follows_the_declared_bounds() -> None:
    assert _compat.in_range("1.66.0", _compat.OPENAI)
    assert not _compat.in_range("1.65.99", _compat.OPENAI)
    assert _compat.in_range("2.99.0", _compat.OPENAI)
    assert not _compat.in_range("3.0.0", _compat.OPENAI)
    assert _compat.in_range("0.40.0", _compat.ANTHROPIC)
    assert not _compat.in_range("1.0.0", _compat.ANTHROPIC)


class Broken(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Makes the import of one module raise ``error``, as a half-installed package does."""

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    @override
    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname == self.name:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    @override
    def exec_module(self, module: ModuleType) -> None:
        raise self.error


def break_import(monkeypatch: pytest.MonkeyPatch, name: str, error: BaseException) -> None:
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Broken(name, error), *sys.meta_path])


class OddImportError(ImportError):
    """An ``ImportError`` subclass, as some packages raise."""


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("module 'typing' has no attribute 'Removed'"),
        RuntimeError("a native extension failed to load"),
        OddImportError("odd"),
        SyntaxError("a half-written file"),
        ValueError("anything"),
    ],
    ids=lambda error: type(error).__name__,
)
@pytest.mark.parametrize(
    ("name", "other"),
    [
        ("openai.resources.chat.completions", "anthropic"),
        ("openai.resources.responses", "anthropic"),
        ("anthropic.resources.messages", "openai"),
        ("anthropic.lib.streaming", "openai"),
        ("openai.resources.beta", "anthropic"),
        ("anthropic.resources.beta.messages", "openai"),
        ("anthropic.lib.bedrock._beta_messages", "openai"),
        ("anthropic.lib.vertex._beta_messages", "openai"),
    ],
)
def test_a_broken_sdk_install_never_raises_out_of_install_and_the_other_sdk_is_recorded(
    name: str,
    other: str,
    error: BaseException,
    tmp_path: Path,
    mock_server: MockServer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    kind = f"{other} ok"
    expected = run_call(kind, mock_server)
    break_import(monkeypatch, name, error)  # (the broken SDK's own calls are not our business)
    with caplog.at_level(logging.WARNING, logger="dagnam_record"):
        result = dagnam_record.install(dir=str(tmp_path / "t"))
    assert result is True  # the other SDK is fine, so it is patched
    said = f"cannot import {name} ({type(error).__name__})" in caplog.text
    assert said is not isinstance(
        error, ImportError
    )  # a missing module is silent, a broken one is not
    assert run_call(kind, mock_server) == expected
    dagnam_record.record_call(provider="other", request={"a": 1}, response={"b": 2})
    lines, _ = lines_and_status(tmp_path / "t")
    assert {line["provider"] for line in lines} == {other, "other"}


def test_a_resource_the_sdk_no_longer_has_is_skipped_and_the_rest_is_recorded(
    tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(Completions, "parse")
    monkeypatch.setitem(sys.modules, "openai.resources.responses", None)
    monkeypatch.delattr("anthropic.resources.messages.Messages.stream")
    assert dagnam_record.install(dir=str(tmp_path / "t")) is True
    run_call("openai ok", mock_server)
    lines, _ = lines_and_status(tmp_path / "t")
    assert [line["provider"] for line in lines] == ["openai"]


def test_an_sdk_whose_every_known_resource_is_gone_is_simply_not_patched(
    tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("openai.resources.chat.completions", "openai.resources.responses"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "openai.resources.beta", None)
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", None)
    monkeypatch.setitem(sys.modules, "anthropic.resources.beta.messages", None)
    monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock._beta_messages", None)
    monkeypatch.setitem(sys.modules, "anthropic.lib.vertex._beta_messages", None)
    assert dagnam_record.install(dir=str(tmp_path / "t")) is False
    assert run_call("openai ok", mock_server)["choices"][0]["message"]["content"] == "Hello there"


def test_an_sdk_module_whose_attribute_lookup_raises_does_not_break_install(
    tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(name: str) -> object:
        raise RuntimeError("a lazy attribute that fails")

    monkeypatch.delitem(vars(openai), "__version__")
    monkeypatch.setitem(vars(openai), "__getattr__", explode)
    assert dagnam_record.install(dir=str(tmp_path / "t")) is True
    run_call("openai ok", mock_server)
    lines, status = lines_and_status(tmp_path / "t")
    assert status["sdk_versions"]["openai"] == _compat.UNKNOWN
    assert "request" not in lines[0]  # no readable version: norm only


def test_a_patched_call_is_norm_only_from_the_first_call_without_install_noting_the_version(
    tmp_path: Path, mock_server: MockServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No call is recorded raw in the gap between the patch going live and the version noted."""
    monkeypatch.setattr(openai, "__version__", "0.28.1")
    writer = Writer(str(tmp_path / "t"), start=False)
    assert _openai.install(Recorder(writer))  # the patch only: install() has not noted anything
    try:
        run_call("openai ok", mock_server)
        writer.close()
    finally:
        _openai.uninstall()
    (line,) = [
        json.loads(text)
        for path in (tmp_path / "t").glob("*.jsonl")
        for text in path.read_text(encoding="utf-8").splitlines()
    ]
    assert "request" not in line
    assert line["norm"]["response"] == "Hello there"


@pytest.mark.parametrize("version", ["0.28.1", "3.0.0", "absent"])
def test_record_call_does_not_follow_the_installed_sdk_version(
    version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if version == "absent":
        monkeypatch.setitem(sys.modules, "openai", None)
    else:
        monkeypatch.setattr(openai, "__version__", version)
    dagnam_record.install(dir=str(tmp_path / "t"))
    dagnam_record.record_call(
        provider="openai",
        request={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
        response={"model": "m", "choices": [{"message": {"role": "assistant", "content": "Yo"}}]},
    )
    (line,) = lines_and_status(tmp_path / "t")[0]
    assert line["request"]["messages"][0]["content"] == "Hi"  # the caller's bodies: kept as given
    assert line["response"]["choices"][0]["message"]["content"] == "Yo"
