"""Shared helpers for the ``dagnam-record`` tests.

A hand-moved clock, a writer the tests can look inside, a scripted queue and a strict
re-implementation of the envelope contract.

Test modules receive these as fixtures (never by importing this module), and annotate them as
``Any`` because a fixture's class cannot be named across modules.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import queue
import re
import threading
from typing import Any, Callable, Dict, Iterator, List, NoReturn, Optional, Tuple

from _mock_server import MockServer
from _openai_calls import API_KEY, ORGANIZATION
import anthropic
import httpx
import openai
import pytest

from dagnam_record import _anthropic, _openai
from dagnam_record._recorder import Recorder
from dagnam_record._writer import Writer

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
JS_SAFE = 2**53 - 1
MAX_LINE_BYTES = 1_000_000
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")


CHAT_REQUEST: Dict[str, Any] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": [{"type": "text", "text": "No jokes."}]},
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}],
        },
        {"role": "user", "content": None},
        "not a message",
        {"content": "no role"},
    ],
}
CHAT_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "gpt-4o-mini-2024",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": 1}'},
                    },
                    {"type": "function", "function": {"name": "other", "arguments": {"a": 1}}},
                    {"type": "function", "function": {"arguments": "{}"}},
                    {"type": "function", "function": "bad"},
                    "bad",
                ],
            }
        }
    ],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        """The current time."""
        return self.now


class Probe(Writer):
    """A writer the tests can look inside, from a subclass, so no test touches a private name."""

    def drain(self) -> None:
        """Write everything queued."""
        self._drain()

    def loop(self, source: Any) -> None:
        """Run the thread's loop over ``source`` in the calling thread."""
        self._run(source)

    def fd(self) -> Optional[int]:
        """The open descriptor, if any."""
        return self._fd

    def alive(self) -> bool:
        """Whether the writer thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def thread(self) -> Optional[threading.Thread]:
        """The writer thread, if one was started."""
        return self._thread

    def pending(self) -> Any:
        """The queue the writer reads."""
        return self._queue

    def gate(self) -> Any:
        """The lock that submit and close take around the closed flag."""
        return self._gate

    def swap_queue(self, replacement: Any) -> None:
        """Use another queue."""
        self._queue = replacement

    def real_write(self) -> Callable[[str], None]:
        """The writer's own write, before a test replaces it."""
        return self._write


class ScriptedQueue:
    """A queue whose ``get`` plays back a script: ``queue.Empty`` means an idle timeout."""

    def __init__(self, script: List[Any]) -> None:
        self.script = script

    def get(self, timeout: Optional[float] = None) -> Optional[str]:
        """The next scripted item."""
        item = self.script.pop(0)
        if item is queue.Empty:
            raise queue.Empty
        return item

    def get_nowait(self) -> Optional[str]:
        """Nothing is left once the script has run."""
        raise queue.Empty


@pytest.fixture
def make(tmp_path: Path) -> Iterator[Callable[..., Probe]]:
    """Build writers (unstarted unless ``start=True``) and close them all afterwards."""
    created: List[Probe] = []

    def factory(directory: Optional[Path] = None, **kwargs: Any) -> Probe:
        kwargs.setdefault("start", False)
        kwargs.setdefault("clock", Clock())
        writer = Probe(str(directory or tmp_path / "traces"), **kwargs)
        created.append(writer)
        return writer

    yield factory
    for writer in created:
        writer.close(timeout=5)


@pytest.fixture
def new_clock() -> Any:
    """The ``Clock`` class."""
    return Clock


@pytest.fixture
def scripted_queue() -> Any:
    """The ``ScriptedQueue`` class."""
    return ScriptedQueue


@pytest.fixture
def chat_bodies() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A Chat Completions request and response with a system turn, text parts and tool calls."""
    return CHAT_REQUEST, CHAT_RESPONSE


@pytest.fixture
def validate_line() -> Callable[[str], None]:
    """``strict_validate``."""
    return strict_validate


def _reject(message: str) -> NoReturn:
    raise ValueError(message)


def _reject_constant(name: str) -> NoReturn:
    _reject(f"{name} is not valid JSON")


def _check_json(root: object) -> None:
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, float):
            if not math.isfinite(value):
                _reject("numbers must be finite")
        elif isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    _reject("object keys must be strings")
                key.encode("utf-8")
                stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)
        elif value is not None and not isinstance(value, int):
            _reject("values must be JSON values")


def _is_one(value: object) -> bool:
    """The integer 1: not 1.0, not True (the real parser checks ``int`` and not ``bool``)."""
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def _natural(obj: Any, key: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= JS_SAFE:
        _reject(f"{key} must be a non-negative integer a JavaScript reader can hold")
    return value


def _text(obj: Any, key: str, allow_empty: bool = False) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not (value or allow_empty):
        _reject(f"{key} must be a string")
    return value


def _check_norm(norm: Any) -> None:
    if not isinstance(norm, dict):
        _reject("norm must be an object")
    _text(norm, "model")
    if norm.get("system") is not None:
        _text(norm, "system", allow_empty=True)
    _text(norm, "response", allow_empty=True)
    if not isinstance(norm.get("messages"), list) or not isinstance(norm.get("tool_calls"), list):
        _reject("norm.messages and norm.tool_calls must be lists")
    for message in norm["messages"]:
        if not isinstance(message, dict):
            _reject("a message must be an object")
        _text(message, "role", allow_empty=True)
        _text(message, "content", allow_empty=True)
    for call in norm["tool_calls"]:
        function = call.get("function") if isinstance(call, dict) else None
        if call.get("type") != "function" or not isinstance(function, dict):
            _reject("a tool call must be a function call")
        _text(function, "name", allow_empty=True)
        _text(function, "arguments", allow_empty=True)
    for key in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens"):
        _natural(norm, key)
    if not isinstance(norm.get("has_media"), bool):
        _reject("norm.has_media must be a boolean")


def strict_validate(line: str) -> None:
    """Raise ``ValueError`` unless ``line`` is an envelope v1 line, by the contract's rules.

    A stdlib re-implementation (this package may not import the contract), a little stricter:
    every count and the latency must also fit a JavaScript number exactly.
    """
    if len(line.encode("utf-8")) > MAX_LINE_BYTES:
        _reject("the line is too long")
    obj = json.loads(line, parse_constant=_reject_constant)
    if not isinstance(obj, dict):
        _reject("an envelope is an object")
    _check_json(obj)
    if not _is_one(obj.get("v")):
        _reject("v must be the integer 1")
    _text(obj, "trace_id")
    if not _TS.fullmatch(_text(obj, "ts")):
        _reject("ts must be UTC RFC 3339 ending in Z")
    datetime.fromisoformat(obj["ts"])
    if obj.get("provider") not in ("openai", "anthropic", "other"):
        _reject("unknown provider")
    if obj.get("endpoint") not in ("chat.completions", "responses", "messages", "custom"):
        _reject("unknown endpoint")
    latency = obj.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        _reject("latency_ms must be a number")
    if not math.isfinite(latency) or not 0 <= latency <= JS_SAFE:
        _reject("latency_ms must be finite, non-negative and exactly representable")
    if math.copysign(1.0, latency) < 0:
        _reject("latency_ms must not be negative zero")
    if not 100 <= _natural(obj, "status") <= 599:
        _reject("status must be an HTTP status")
    request, response = obj.get("request"), obj.get("response")
    if (request is None) != (response is None):
        _reject("request and response go together")
    if request is not None and not (isinstance(request, dict) and isinstance(response, dict)):
        _reject("request and response must be objects")
    if obj.get("norm") is not None:
        if not _is_one(obj.get("norm_v")):
            _reject("norm_v must be the integer 1")
        _check_norm(obj["norm"])
    elif request is None:
        _reject("a line needs request and response, or norm")


class Rig:
    """The OpenAI patch installed over a writer, plus real clients aimed at the mock server."""

    def __init__(
        self, server: MockServer, writer: Probe, recorder: Recorder, directory: Path
    ) -> None:
        self.server = server
        self.writer = writer
        self.recorder = recorder
        self.directory = directory

    def client(self, **kwargs: Any) -> openai.OpenAI:
        """A real client that talks to the mock server (no retries, no proxy from the env)."""
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("base_url", self.server.url)
        return openai.OpenAI(
            api_key=API_KEY,
            organization=ORGANIZATION,
            http_client=httpx.Client(trust_env=False),
            **kwargs,
        )

    def aclient(self, **kwargs: Any) -> openai.AsyncOpenAI:
        """The async twin of ``client``."""
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("base_url", self.server.url)
        return openai.AsyncOpenAI(
            api_key=API_KEY,
            organization=ORGANIZATION,
            http_client=httpx.AsyncClient(trust_env=False),
            **kwargs,
        )

    def raw(self) -> List[str]:
        """Every recorded line (each checked against the strict contract), after a drain."""
        self.writer.drain()
        found: List[str] = []
        for path in sorted(self.directory.glob("*.jsonl")):
            found.extend(path.read_text(encoding="utf-8").splitlines())
        for line in found:
            strict_validate(line)
        return found

    def lines(self) -> List[Dict[str, Any]]:
        """Every recorded line, parsed."""
        return [json.loads(line) for line in self.raw()]


@pytest.fixture
def mock_server() -> Iterator[MockServer]:
    """A running mock OpenAI server, stopped afterwards."""
    server = MockServer()
    yield server
    server.close()


@pytest.fixture
def rig(make: Callable[..., Probe], mock_server: MockServer, tmp_path: Path) -> Iterator[Rig]:
    """The OpenAI patch installed over an unstarted writer; removed afterwards."""
    writer = make()
    recorder = Recorder(writer)
    assert _openai.install(recorder)
    yield Rig(mock_server, writer, recorder, tmp_path / "traces")
    _openai.uninstall()


class AnthropicRig(Rig):
    """The Anthropic patch installed over a writer, plus real clients aimed at the mock server."""

    def anthropic(self, **kwargs: Any) -> anthropic.Anthropic:
        """A real client that talks to the mock server (no retries, no proxy from the env)."""
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("base_url", self.server.root)
        return anthropic.Anthropic(
            api_key=API_KEY,
            http_client=httpx.Client(trust_env=False),
            **kwargs,
        )

    def aanthropic(self, **kwargs: Any) -> anthropic.AsyncAnthropic:
        """The async twin of ``anthropic``."""
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("base_url", self.server.root)
        return anthropic.AsyncAnthropic(
            api_key=API_KEY,
            http_client=httpx.AsyncClient(trust_env=False),
            **kwargs,
        )


@pytest.fixture
def arig(
    make: Callable[..., Probe], mock_server: MockServer, tmp_path: Path
) -> Iterator[AnthropicRig]:
    """The Anthropic patch installed over an unstarted writer; removed afterwards."""
    writer = make()
    recorder = Recorder(writer)
    assert _anthropic.install(recorder)
    yield AnthropicRig(mock_server, writer, recorder, tmp_path / "traces")
    _anthropic.uninstall()
