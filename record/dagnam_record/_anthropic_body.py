"""Request and response bodies of Anthropic calls as plain JSON, without credentials or media payloads."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from dagnam_record._envelope import omit_media
from dagnam_record._recorder import media_omitted, to_plain

Body = Dict[str, Any]
TRANSPORT: FrozenSet[str] = frozenset(
    {"extra_headers", "extra_query", "extra_body", "timeout", "api_key", "auth_token"}
)
REDACTED = "<redacted>"
_CREDIT_FIELD = "fallback_credit_token"  # a bearer code for a retry's billing
_SERVER_FIELD = "authorization_token"  # what an MCP server definition carries


def _redact(body: Body) -> None:
    """Replace the secrets a request body can carry (the SDK sends them, the record must not)."""
    if _CREDIT_FIELD in body:
        body[_CREDIT_FIELD] = REDACTED
    servers = body.get("mcp_servers")
    for server in servers if isinstance(servers, list) else []:
        if isinstance(server, dict) and _SERVER_FIELD in server:
            server[_SERVER_FIELD] = REDACTED


def request_body(kwargs: Dict[str, Any]) -> Body:
    """The call's JSON body parameters: its keyword arguments minus the transport ones."""
    body = to_plain({key: value for key, value in kwargs.items() if key not in TRANSPORT})
    _redact(body)
    return body


def _omit_base64(value: Any) -> Any:
    """``value`` with the data of every base64 source (an image or file a tool made) replaced by its size."""
    if isinstance(value, dict):
        if value.get("type") == "base64" and isinstance(value.get("data"), str):
            return dict(value, data=media_omitted(value["data"]))
        return {key: _omit_base64(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_omit_base64(item) for item in value]
    return value


def response_body(result: Any) -> Body:
    """A response as plain JSON with its media payloads replaced by their size."""
    body = to_plain(result)
    if not isinstance(body, dict):
        raise ValueError("the response is not a model (a raw or streaming-response call)")
    return omit_media(_omit_base64(body))
