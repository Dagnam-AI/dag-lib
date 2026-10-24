"""What every SDK patch shares: turn a call into one line, fail open, and honour the kill switch.

A patch hands over the request and response bodies as plain JSON data (``to_plain`` makes SDK
objects into that) and the recorder builds the envelope line and queues it. Nothing here reads a
header or a key: a patch passes the call's JSON body parameters and nothing else.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
import hashlib
import os
import time
from typing import Any, Callable, Dict, Optional, Set
import uuid

from dagnam_record._envelope import LineTooLargeError, build_line
from dagnam_record._redact import apply
from dagnam_record._writer import Writer

KILL_SWITCH = "DAGNAM_RECORD"
MEDIA_OMITTED = "media_omitted"
_MAX_DEPTH = 64
MAX_NODES = 250_000
"""Parts one body may have: a 1 MB line cannot hold many more than this."""
_UNSET_NAMES = frozenset({"NotGiven", "Omit"})  # the SDKs' "argument not passed" sentinels


OFF_VALUES = frozenset({"0", "false", "off", "no", "disable", "disabled"})


def recording_disabled() -> bool:
    """The kill switch: ``DAGNAM_RECORD`` switches the recorder off.

    Off is one of ``0``, ``false``, ``off``, ``no``, ``disable``, ``disabled``, in any case and
    with spaces around it. Everything else leaves the recorder on: unset, empty, ``1``, ``00``,
    ``yes``, ``true``, ``on`` or any other text.
    """
    return os.environ.get(KILL_SWITCH, "").strip().lower() in OFF_VALUES


def sampled(trace_id: str, rate: float) -> bool:
    """Whether the call with this trace id is kept at this ``rate`` in ``(0, 1]``.

    A hash of the id decides, not a random draw, so the same id always gets the same answer
    (a call that is seen twice is kept twice or dropped twice, never half-recorded), and a
    larger rate keeps every call a smaller one kept.
    """
    digest = hashlib.sha256(trace_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


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


class Walk:
    """The state of one conversion: how many nodes were visited and which containers are open."""

    def __init__(self) -> None:
        self.nodes = 0
        self.open: Set[int] = set()

    def visit(self) -> None:
        """Count one node; past ``MAX_NODES`` the body is too big to record."""
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise LineTooLargeError("the body has too many parts to record")


def _class_schema(value: type) -> Any:
    """A pydantic model class as the JSON schema it stands for; any other class by name."""
    method = getattr(value, "model_json_schema", None)
    schema = _attempt(method, {}) if callable(method) else None
    if isinstance(schema, dict):
        return {"type": "json_schema", "name": value.__name__, "schema": _plain(schema, Walk(), 0)}
    return f"<class {value.__name__}>"


def _plain(value: Any, walk: Walk, depth: int) -> Any:
    walk.visit()
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
    return _plain(value, Walk(), 0)


class Recorder:
    """Builds lines from plain bodies and hands them to the writer, counting what it drops.

    ``sample`` keeps that share of calls (see ``sampled``); ``redact`` is applied to the request
    and to the response text before the line is built (see ``_redact``).
    """

    def __init__(
        self,
        writer: Writer,
        *,
        no_raw: bool = False,
        sample: float = 1.0,
        redact: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.writer = writer
        self.no_raw = no_raw
        self.sample = sample
        self.redact = redact

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
        self.record(
            provider,
            endpoint,
            request,
            response,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def record(
        self,
        provider: str,
        endpoint: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        *,
        status: Optional[int],
        latency_ms: float,
    ) -> None:
        """Build one line for a finished call with a known latency and queue it (or skip it).

        Raises ``ValueError`` for a call that cannot be recorded; callers go through ``fail_open``.
        """
        trace_id = str(uuid.uuid4())
        if not sampled(trace_id, self.sample):
            return
        if self.redact is not None:
            request, response = apply(self.redact, request), apply(self.redact, response)
        self.writer.submit(
            build_line(
                provider=provider,
                endpoint=endpoint,
                request=request,
                response=response,
                latency_ms=latency_ms,
                status=status,
                trace_id=trace_id,
                no_raw=self.no_raw,
            )
        )
