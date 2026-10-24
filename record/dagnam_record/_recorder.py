"""What every SDK patch shares: turn a call into one line, fail open, and honour the kill switch.

A patch hands over the request and response bodies as plain JSON data (``to_plain`` makes SDK
objects into that) and the recorder builds the envelope line and queues it. Nothing here reads a
header or a key: a patch passes the call's JSON body parameters and nothing else.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
import os
import time
from typing import Any, Callable, Dict, Optional, Set

from dagnam_record._envelope import LineTooLargeError, build_line
from dagnam_record._writer import Writer

KILL_SWITCH = "DAGNAM_RECORD"
MEDIA_OMITTED = "media_omitted"
_MAX_DEPTH = 64
MAX_NODES = 250_000
"""Parts one body may have: a 1 MB line cannot hold many more than this."""
_UNSET_NAMES = frozenset({"NotGiven", "Omit"})  # the SDKs' "argument not passed" sentinels


def recording_disabled() -> bool:
    """``DAGNAM_RECORD=0`` switches the recorder off."""
    return os.environ.get(KILL_SWITCH) == "0"


def media_omitted(payload: str) -> Dict[str, Any]:
    """The stand-in for a base64 payload: its size, so the line stays small and valid."""
    return {"type": MEDIA_OMITTED, "bytes": len(payload.encode("utf-8"))}


def _attempt(method: Callable[..., Any], arguments: Dict[str, Any]) -> Any:
    """What ``method`` returns, or ``None`` when it raises (an unreadable object is a placeholder)."""
    try:
        return method(**arguments)
    except Exception:
        return None


def _dump(value: Any) -> Optional[Dict[str, Any]]:
    """A plain dict from an SDK object (pydantic v2 or v1, or anything with ``to_dict``)."""
    attempts: Any = (
        ("model_dump", {"mode": "json", "warnings": False, "exclude_unset": True}),
        ("model_dump", {"mode": "json"}),
        ("to_dict", {}),
        ("dict", {}),
    )
    for name, arguments in attempts:
        method = getattr(value, name, None)
        dumped = _attempt(method, arguments) if callable(method) else None
        if isinstance(dumped, dict):
            return dumped
    return None


class _Walk:
    """The state of one conversion: how many nodes were visited and which containers are open."""

    def __init__(self) -> None:
        self.nodes = 0
        self.open: Set[int] = set()


def _class_schema(value: type) -> Any:
    """A pydantic model class as the JSON schema it stands for; any other class by name."""
    method = getattr(value, "model_json_schema", None)
    schema = _attempt(method, {}) if callable(method) else None
    if isinstance(schema, dict):
        return {"type": "json_schema", "name": value.__name__, "schema": _plain(schema, _Walk(), 0)}
    return f"<class {value.__name__}>"


def _plain(value: Any, walk: _Walk, depth: int) -> Any:
    walk.nodes += 1
    if walk.nodes > MAX_NODES:
        raise LineTooLargeError("the body has too many parts to record")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= _MAX_DEPTH:
        return "<too deep>"
    if isinstance(value, (dict, list, tuple)):
        if id(value) in walk.open:
            raise ValueError("circular reference in the body")
        walk.open.add(id(value))
        try:
            if isinstance(value, dict):
                return {
                    str(key): _plain(item, walk, depth + 1)
                    for key, item in value.items()
                    if type(item).__name__ not in _UNSET_NAMES
                }
            return [_plain(item, walk, depth + 1) for item in value]
        finally:
            walk.open.discard(id(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": MEDIA_OMITTED, "bytes": len(value)}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _plain(value.value, walk, depth + 1)
    if isinstance(value, type):
        return _class_schema(value)
    dumped = _dump(value)
    return _plain(dumped, walk, depth + 1) if dumped is not None else f"<{type(value).__name__}>"


def to_plain(value: Any) -> Any:
    """A copy of ``value`` made only of JSON types; it raises only ``ValueError``.

    SDK objects become dicts (what the server sent, unset fields left out), ``bytes`` become a
    size marker, dates become ISO text, a pydantic model class becomes its JSON schema, "not
    given" sentinels are dropped and anything else is named by its class, never printed (so no
    secret in an object's repr can leak).

    The walk is bounded: a container that contains itself raises ``ValueError``, and a body with
    more than ``MAX_NODES`` parts (a shared object expanded at every place it appears can make
    a small structure enormous) raises ``LineTooLargeError``, so no input costs more than a
    constant number of steps.
    """
    return _plain(value, _Walk(), 0)


class Recorder:
    """Builds lines from plain bodies and hands them to the writer, counting what it drops."""

    def __init__(self, writer: Writer, *, no_raw: bool = False) -> None:
        self.writer = writer
        self.no_raw = no_raw

    def fail_open(self, step: Callable[..., None], *arguments: Any) -> bool:
        """Run ``step``; any error is counted, never raised. True when the step ran cleanly."""
        try:
            step(*arguments)
            return True
        except LineTooLargeError:
            self.writer.count("dropped_large")
        except Exception:  # the recorder must never hurt the call it describes
            self.writer.count("dropped_error")
        return False

    def emit(
        self,
        provider: str,
        endpoint: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        *,
        status: Optional[int],
        started: float,
    ) -> None:
        """Build one line for a finished call and queue it. ``started`` is a ``perf_counter``."""
        self.writer.submit(
            build_line(
                provider=provider,
                endpoint=endpoint,
                request=request,
                response=response,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                status=status,
                no_raw=self.no_raw,
            )
        )
