"""The platform's insufficient-credits refusal (402), read from its decoded body.

The body is ``{"error": "insufficient_credits", "message": <a complete sentence>,
"required_credits": int, "available_credits": int (omitted when the caller is not the
account's owner), "next_steps": [<token>, ...], ...}``. A platform that predates it sends only
the ``detail`` copy of the same fields (``user_message`` for the sentence), which is read when the
top level carries no marker. A body without the marker, or without a numeric
``required_credits``, is not read here: it stays the plan-limit shape the caller maps.
"""

from __future__ import annotations

from dagnam._core.exceptions import InsufficientCreditsError
from dagnam._types import JsonObject

INSUFFICIENT_CREDITS = "insufficient_credits"
"""The ``error`` a credit refusal carries."""
FALLBACK_MESSAGE = "You don't have enough credits for this."
"""Said only when a refusal arrives without its own message."""


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def credit_refusal(data: JsonObject | None) -> InsufficientCreditsError | None:
    """The typed error for a decoded 402 body, or ``None`` when it is not a credit refusal."""
    if data is None:
        return None
    detail = data.get("detail")
    source = data if data.get("error") == INSUFFICIENT_CREDITS else detail
    if not isinstance(source, dict) or source.get("error") != INSUFFICIENT_CREDITS:
        return None
    required = _count(source.get("required_credits"))
    if required is None:
        return None
    message = source.get("message") or source.get("user_message")
    steps = data.get("next_steps")
    return InsufficientCreditsError(
        message if isinstance(message, str) and message else FALLBACK_MESSAGE,
        required_credits=required,
        available_credits=_count(source.get("available_credits")),
        next_steps=tuple(s for s in steps if isinstance(s, str)) if isinstance(steps, list) else (),
    )
