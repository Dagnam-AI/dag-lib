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
from typing import Any, Callable, Iterator, List, NoReturn, Optional

import pytest

from dagnam_record._writer import Writer

T0 = 1_791_797_400.0  # 2026-10-12T09:30:00Z
JS_SAFE = 2**53 - 1
MAX_LINE_BYTES = 1_000_000
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")


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
    if obj.get("v") != 1 or isinstance(obj.get("v"), bool):
        _reject("v must be 1")
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
        if obj.get("norm_v") != 1:
            _reject("norm_v must be 1")
        _check_norm(obj["norm"])
    elif request is None:
        _reject("a line needs request and response, or norm")
