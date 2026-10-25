"""Fold the events of an OpenAI stream into the response a non-stream call would have returned.

An assembler takes events with ``add``; ``complete`` says whether the stream reached its own
end marker (a finished answer) and ``result()`` is then the response body, ``status`` the status
to record it with.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dagnam_record._openai_body import Body, response_body
from dagnam_record._recorder import MEDIA_OMITTED, to_plain
from dagnam_record._stream import get

_TERMINAL = ("response.completed", "response.incomplete", "response.failed")


def _new_call() -> Body:
    return {"id": None, "name": "", "arguments": ""}


class ChatAssembler:
    """Fold ``chat.completion.chunk`` events into the message a non-stream call returns."""

    status = 200

    def __init__(self) -> None:
        self.size = 0  # characters of text kept so far (content, tool arguments, transcripts)
        self._head: Body = {}
        self._choices: Dict[int, Body] = {}
        self._usage: Any = None

    @property
    def complete(self) -> bool:
        """True once every choice that appeared has a ``finish_reason``."""
        return bool(self._choices) and all(slot["finish"] for slot in self._choices.values())

    def add(self, chunk: Any) -> None:
        """Take one chunk."""
        for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
            if key not in self._head and get(chunk, key) is not None:
                self._head[key] = get(chunk, key)
        if get(chunk, "usage") is not None:
            self._usage = get(chunk, "usage")
        for choice in get(chunk, "choices") or []:
            self._choice(choice)

    def _choice(self, choice: Any) -> None:
        index = get(choice, "index") or 0
        slot = self._choices.setdefault(
            index,
            {"role": None, "content": [], "refusal": [], "calls": {}, "last": None, "finish": None},
        )
        if get(choice, "finish_reason"):
            slot["finish"] = get(choice, "finish_reason")
        delta = get(choice, "delta")
        if get(delta, "role"):
            slot["role"] = get(delta, "role")
        for key in ("content", "refusal"):
            if isinstance(get(delta, key), str):
                slot[key].append(get(delta, key))
                self.size += len(get(delta, key))
        for call in get(delta, "tool_calls") or []:
            self.size += self._add_to(slot["calls"], self._call_index(slot, call), call)
        if get(delta, "function_call") is not None:
            self.size += self._add_to(slot["calls"], -1, {"function": get(delta, "function_call")})
        if get(delta, "audio") is not None:
            audio = slot.setdefault("audio", {"id": None, "transcript": [], "bytes": 0})
            audio["id"] = get(get(delta, "audio"), "id") or audio["id"]
            audio["transcript"].append(get(get(delta, "audio"), "transcript") or "")
            self.size += len(audio["transcript"][-1])
            audio["bytes"] += len(get(get(delta, "audio"), "data") or "")

    @staticmethod
    def _call_index(slot: Body, call: Any) -> Any:
        """Which tool call a delta belongs to.

        ``index`` when the server sends one. Servers that do not send it (OpenAI-compatible
        ones) are followed by the ids: a delta with an id other than the open call's, or with a
        function name when the open call already has one, starts a new call; anything else
        continues the open call.
        """
        index = get(call, "index")
        if index is None:
            calls: Dict[int, Body] = slot["calls"]
            current = calls.get(slot["last"])  # None before the first call
            call_id = get(call, "id")
            name = get(get(call, "function"), "name")
            new_id = bool(call_id and current and current["id"] and call_id != current["id"])
            new_name = bool(name and current and current["name"])
            if current is None or new_id or new_name:
                index = max([key for key in calls if key >= 0], default=-1) + 1
            else:
                index = slot["last"]
        slot["last"] = index
        return index

    @staticmethod
    def _add_to(calls: Dict[int, Body], index: int, delta: Any) -> int:
        """Add a tool call delta; the characters it added."""
        call = calls.setdefault(index, _new_call())
        call["id"] = get(delta, "id") or call["id"]
        function = get(delta, "function")
        name, arguments = get(function, "name") or "", get(function, "arguments") or ""
        call["name"] += name
        call["arguments"] += arguments
        return len(name) + len(arguments)

    def result(self) -> Body:
        """The assembled response body."""
        body = dict(self._head, object="chat.completion")
        body["choices"] = [
            self._message(index, self._choices[index]) for index in sorted(self._choices)
        ]
        if self._usage is not None:
            body["usage"] = to_plain(self._usage)
        return body

    @staticmethod
    def _message(index: int, slot: Body) -> Body:
        message: Body = {
            "role": slot["role"] or "assistant",
            "content": "".join(slot["content"]) if slot["content"] else None,
        }
        if slot["refusal"]:
            message["refusal"] = "".join(slot["refusal"])
        calls: List[Body] = [slot["calls"][key] for key in sorted(slot["calls"]) if key >= 0]
        if calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in calls
            ]
        if -1 in slot["calls"]:
            message["function_call"] = {
                "name": slot["calls"][-1]["name"],
                "arguments": slot["calls"][-1]["arguments"],
            }
        if "audio" in slot:
            audio = slot["audio"]
            message["audio"] = {
                "id": audio["id"],
                "transcript": "".join(audio["transcript"]),
                "data": {"type": MEDIA_OMITTED, "bytes": audio["bytes"]},
            }
        return {"index": index, "message": message, "finish_reason": slot["finish"]}


class ResponsesAssembler:
    """Keep the terminal ``response.*`` event of a Responses stream: it carries the whole response."""

    size = 0  # nothing builds up: only the last event is kept

    def __init__(self) -> None:
        self._kind: Optional[str] = None
        self._response: Any = None

    @property
    def complete(self) -> bool:
        """True once a terminal event (completed, incomplete or failed) has been seen."""
        return self._kind is not None

    @property
    def status(self) -> Optional[int]:
        """200, or ``None`` (no usable response) when the stream ended in ``response.failed``."""
        return None if self._kind == "response.failed" else 200

    def add(self, event: Any) -> None:
        """Take one event."""
        if get(event, "type") in _TERMINAL:
            self._kind, self._response = get(event, "type"), get(event, "response")

    def result(self) -> Body:
        """The final response body, or the failure for ``response.failed``."""
        if self._kind == "response.failed":
            return {"error": {"type": self._kind}}
        return response_body(self._response)
