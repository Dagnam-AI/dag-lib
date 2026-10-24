"""Small helpers shared by the Anthropic patch tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List

if TYPE_CHECKING:
    import anthropic

MESSAGES: List[Dict[str, Any]] = [{"role": "user", "content": "Hi"}]
ARGS: Dict[str, Any] = {"model": "claude-sonnet-4-5", "max_tokens": 64, "messages": MESSAGES}


def run(call: Callable[[anthropic.AsyncAnthropic], Awaitable[Any]], rig: Any) -> Any:
    """Run ``call`` with an async client aimed at the mock server, closing it afterwards."""

    async def main() -> Any:
        async with rig.aanthropic() as client:
            return await call(client)

    return asyncio.run(main())


def unpatched(method: Any) -> Any:
    """The SDK's own function behind a patched one (through the SDK's own decorators)."""
    return method.__wrapped__
