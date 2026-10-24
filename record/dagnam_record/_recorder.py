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
from typing import Any, Callable, Dict, Optional

from dagnam_record._envelope import LineTooLargeError, build_line
from dagnam_record._writer import Writer

KILL_SWITCH = "DAGNAM_RECORD"
MEDIA_OMITTED = "media_omitted"
_MAX_DEPTH = 64
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


def to_plain(value: Any, depth: int = 0) -> Any:
    """A copy of ``value`` made only of JSON types; it never raises.

    SDK objects become dicts (what the server sent, unset fields left out), ``bytes`` become a
    size marker, dates become ISO text, "not given" sentinels are dropped and anything else is
    named by its class, never printed (so no secret in an object's repr can leak).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= _MAX_DEPTH:
        return "<too deep>"
    if isinstance(value, dict):
        return {
            str(key): to_plain(item, depth + 1)
            for key, item in value.items()
            if type(item).__name__ not in _UNSET_NAMES
        }
    if isinstance(value, (list, tuple)):
        return [to_plain(item, depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": MEDIA_OMITTED, "bytes": len(value)}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_plain(value.value, depth + 1)
    dumped = _dump(value)
    return to_plain(dumped, depth + 1) if dumped is not None else f"<{type(value).__name__}>"


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
