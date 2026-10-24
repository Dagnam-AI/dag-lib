"""The ``redact`` option: a function over the serialized request and response text.

``redact`` is called twice for a call, once with the request as JSON text and once with the
response, before the line (and its ``norm``) is built from what comes back. It must return JSON
text of an object. If it raises, or returns anything else, the call is dropped and counted as
``dropped_error``: a line is never written from a body that was not redacted. ``"auto"`` is the
PII redactor of the optional ``dagnam-contracts`` package (extra ``dagnam-record[redact]``).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Union, cast

AUTO = "auto"
Redactor = Callable[[str], str]
Option = Union[Redactor, str, None]


def check(redact: object) -> Option:
    """``redact`` itself when it is ``None``, ``"auto"`` or a callable; else ``ValueError``."""
    if redact is None or (isinstance(redact, str) and redact == AUTO):
        return redact
    if callable(redact):
        return cast("Redactor", redact)
    raise ValueError('redact must be None, a function from text to text, or "auto"')


def resolve(redact: Option) -> Optional[Redactor]:
    """The function to call, or ``None``; ``"auto"`` needs ``dagnam-record[redact]`` (``ImportError``)."""
    if not isinstance(redact, str):
        return redact
    try:
        # The one runtime-optional import in this package: only for ``redact="auto"``, and
        # only when the extra is installed (checked by tests/structure/test_record_package.py).
        from dagnam_contracts.hygiene import redact_json_text
    except ImportError as exc:
        raise ImportError(
            'redact="auto" needs the redactor from dagnam-contracts: '
            'pip install "dagnam-record[redact]"'
        ) from exc

    def automatic(text: str) -> str:
        return redact_json_text(text)[0]

    return automatic


def apply(redact: Redactor, body: object) -> Dict[str, Any]:
    """``body`` after ``redact`` has seen its JSON text; ``ValueError`` if that is not text or not JSON."""
    if not isinstance(body, dict):
        raise ValueError("a body must be a JSON object")
    # A redactor is user code: what it returns is checked, whatever its annotation says.
    text = cast("object", redact(json.dumps(body, ensure_ascii=False, allow_nan=False)))
    if not isinstance(text, str):
        raise ValueError("redact must return text")
    # Not checked to be an object here: ``build_line`` refuses any body that is not one.
    return cast("Dict[str, Any]", json.loads(text))
