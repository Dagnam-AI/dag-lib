"""Small helpers shared by the OpenAI patch tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    import openai


API_KEY = "sk-proj-must-never-be-recorded-0123"
ORGANIZATION = "org-must-never-be-recorded"


def run(call: Callable[[openai.AsyncOpenAI], Awaitable[Any]], rig: Any) -> Any:
    """Run ``call`` with an async client aimed at the mock server, closing it afterwards."""

    async def main() -> Any:
        async with rig.aclient() as client:
            return await call(client)

    return asyncio.run(main())


def unpatched(method: Any) -> Any:
    """The SDK's own function behind a patched one."""
    return method.__wrapped__
