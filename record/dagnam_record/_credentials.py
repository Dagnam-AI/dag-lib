"""Credentials a request can carry inside its tool definitions, replaced before a line is built.

OpenAI's Responses ``tools`` take an ``mcp`` tool with an ``authorization`` and ``headers``,
Anthropic's ``mcp_servers`` take an ``authorization_token``, and vendors add their own: the SDKs
send these to the server, the record must not hold them. Only the entries of ``tools`` and
``mcp_servers`` are looked at (a message that mentions a "token" is the user's text, not a
credential), by the name of the key, in any case. Every value under ``headers`` goes, and so does
the user info and the query string of a server URL, where tokens are often put.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet

REDACTED = "<redacted>"
_LISTS = ("tools", "mcp_servers")
_SECRET_KEYS: FrozenSet[str] = frozenset(
    {
        "authorization",
        "authorization_token",
        "api_key",
        "access_token",
        "token",
        "secret",
        "client_secret",
        "password",
    }
)
_URL_KEYS = frozenset({"server_url", "url"})
_USER_INFO = re.compile(r"(?<=//)[^/?#@]*@")


def _url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _USER_INFO.sub(REDACTED + "@", value)
    return text.split("?", 1)[0] + "?" + REDACTED if "?" in text else text


def _entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    cleaned: Dict[str, Any] = {}
    for key, value in entry.items():
        name = str(key).lower()
        if name in _SECRET_KEYS and value is not None:
            cleaned[key] = REDACTED
        elif name == "headers" and value is not None:
            cleaned[key] = dict.fromkeys(value, REDACTED) if isinstance(value, dict) else REDACTED
        elif name in _URL_KEYS:
            cleaned[key] = _url(value)
        else:
            cleaned[key] = value
    return cleaned


def hide_credentials(body: Any) -> Any:
    """A copy of the request ``body`` with the credentials of its tool definitions replaced.

    Anything but an object comes back as it is (the line builder refuses it).
    """
    if not isinstance(body, dict):
        return body
    cleaned = dict(body)
    for name in _LISTS:
        entries = cleaned.get(name)
        if isinstance(entries, list):
            cleaned[name] = [_entry(entry) for entry in entries]
    return cleaned
