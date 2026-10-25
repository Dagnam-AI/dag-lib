"""Request and response bodies of OpenAI calls as plain JSON, with media payloads left out."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from dagnam_record._envelope import omit_media
from dagnam_record._recorder import media_omitted, to_plain

Body = Dict[str, Any]
TRANSPORT: FrozenSet[str] = frozenset(
    {"extra_headers", "extra_query", "extra_body", "timeout", "api_key"}
)
_AUDIO_PARTS = frozenset({"output_audio", "audio"})


def _replace_payload(container: Any, key: str) -> None:
    if isinstance(container, dict) and isinstance(container.get(key), str):
        container[key] = media_omitted(container[key])


def _omit_output_media(items: Any) -> None:
    """Swap base64 outputs (generated images, audio) in Responses-style items for their size."""
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            if item.get("type") == "image_generation_call":
                _replace_payload(item, "result")
            parts = item.get("content")
            for part in parts if isinstance(parts, list) else []:
                if isinstance(part, dict) and part.get("type") in _AUDIO_PARTS:
                    _replace_payload(part, "data")


def request_body(kwargs: Dict[str, Any]) -> Body:
    """The call's JSON body parameters: its keyword arguments minus the transport ones."""
    body = to_plain({key: value for key, value in kwargs.items() if key not in TRANSPORT})
    _omit_output_media(body.get("input"))
    return body


def response_body(result: Any) -> Body:
    """A response as plain JSON with its media payloads (base64 audio, images) replaced by size."""
    body = to_plain(result)
    if not isinstance(body, dict):
        raise ValueError("the response is not a model (a raw or streaming-response call)")
    for choice in body.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        _replace_payload(message.get("audio") if isinstance(message, dict) else None, "data")
    _omit_output_media(body.get("output"))
    return omit_media(body)
