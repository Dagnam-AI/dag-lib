"""Fold the events of an Anthropic message stream into the message a non-stream call returns.

Events, as the API documents them: ``message_start`` carries the message (id, model, input
usage) with no content yet; each content block is opened by ``content_block_start`` and filled by
``content_block_delta`` events (``text_delta``, ``input_json_delta`` for tool inputs,
``thinking_delta``, ``signature_delta``, ``citations_delta``); ``message_delta`` carries the stop
reason and the final output usage; ``message_stop`` ends the stream. ``ping`` and event types
this module does not know are ignored. ``complete`` is true once ``message_stop`` has been seen
(or an ``error`` event, which makes the stream a failed call).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dagnam_record._anthropic_body import Body, response_body
from dagnam_record._recorder import to_plain
from dagnam_record._stream import get

ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}
"""The HTTP status the API documents for each error type: a stream that fails after the 200."""


_PIECES = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}


def _index(event: Any) -> Optional[int]:
    index = get(event, "index")
    return index if isinstance(index, int) and not isinstance(index, bool) else None


def _tool_input(pieces: List[str]) -> Body:
    """The JSON object the pieces make; ``{}`` when there are none, or they are cut short."""
    try:
        parsed = json.loads("".join(pieces) or "{}")
    except ValueError:  # e.g. the answer stopped at max_tokens in the middle of a tool call
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MessageAssembler:
    """Fold ``message_start`` .. ``message_stop`` events into one message body."""

    def __init__(self) -> None:
        self._head: Optional[Body] = None  # the message of message_start, no content
        self._usage: Body = {}
        self._blocks: Dict[int, Body] = {}
        self._pieces: Dict[int, Dict[str, List[str]]] = {}  # per block: field -> text pieces
        self._stopped = False
        self._error: Optional[str] = None  # the error type of an ``error`` event

    @property
    def complete(self) -> bool:
        """True once the stream reached its end marker (or failed with an ``error`` event)."""
        return self._error is not None or (self._stopped and self._head is not None)

    @property
    def status(self) -> Optional[int]:
        """200, or the status of the error event (``None``: no usable response, recorded as 599)."""
        return 200 if self._error is None else ERROR_STATUS.get(self._error)

    def add(self, event: Any) -> None:
        """Take one event."""
        kind = get(event, "type")
        if kind == "message_start":
            self._start(get(event, "message"))
        elif kind == "content_block_start":
            self._open(event)
        elif kind == "content_block_delta":
            self._delta(event)
        elif kind == "message_delta":
            self._finish(event)
        elif kind == "message_stop":
            self._stopped = True
        elif kind == "error":
            error_type = get(get(event, "error"), "type")
            self._error = error_type if isinstance(error_type, str) else "error"

    def _start(self, message: Any) -> None:
        head = to_plain(message)
        if isinstance(head, dict):
            self._usage = dict(head.pop("usage", None) or {})
            self._head = head

    def _open(self, event: Any) -> None:
        index, block = _index(event), to_plain(get(event, "content_block"))
        if index is not None and isinstance(block, dict):
            self._blocks[index] = block
            self._pieces.pop(index, None)

    def _delta(self, event: Any) -> None:
        index, delta = _index(event), get(event, "delta")
        block = self._blocks.get(index) if index is not None else None
        if index is None or block is None:  # a delta for a block that was never opened
            return
        kind = get(delta, "type")
        field = _PIECES.get(str(kind))  # the delta's text lives under the key of the same name
        if field is not None and isinstance(get(delta, field), str):
            self._pieces.setdefault(index, {}).setdefault(field, []).append(get(delta, field))
        elif kind == "signature_delta" and isinstance(get(delta, "signature"), str):
            block["signature"] = get(delta, "signature")  # one piece, as the SDK takes it
        elif kind == "citations_delta":
            citations = block.get("citations")
            block["citations"] = (citations if isinstance(citations, list) else []) + [
                to_plain(get(delta, "citation"))
            ]

    def _finish(self, event: Any) -> None:
        delta = to_plain(get(event, "delta"))
        if self._head is not None and isinstance(delta, dict):
            self._head.update(delta)  # stop_reason, stop_sequence, anything newer
        usage = to_plain(get(event, "usage"))
        if isinstance(usage, dict):  # cumulative counts: the latest non-null value wins
            self._usage.update({key: value for key, value in usage.items() if value is not None})

    def result(self) -> Body:
        """The assembled message body, or the failure for an ``error`` event."""
        if self._error is not None:
            return {"error": {"type": self._error}}
        body = dict(self._head or {})
        body["content"] = [self._block(index) for index in sorted(self._blocks)]
        body["usage"] = self._usage
        return response_body(body)

    def _block(self, index: int) -> Body:
        block = dict(self._blocks[index])
        pieces = self._pieces.get(index, {})
        for field in ("text", "thinking"):
            if field in pieces:
                block[field] = str(block.get(field) or "") + "".join(pieces[field])
        if "partial_json" in pieces:
            block["input"] = _tool_input(pieces["partial_json"])
        return block
