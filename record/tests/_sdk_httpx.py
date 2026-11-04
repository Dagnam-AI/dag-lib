"""The HTTP library the installed ``openai`` and ``anthropic`` use, under the name ``httpx``.

``openai`` 3 and ``anthropic`` 2 moved from ``httpx`` to its fork ``httpx2`` and reject a client of
the old package, so a test that builds a client or a mock transport takes the module from here.
"""

from __future__ import annotations

try:
    import httpx2 as httpx  # pyright: ignore[reportMissingImports]  # installed with openai 3 only
except ImportError:  # the SDKs of the main gate (openai 2, anthropic 0.x) still use httpx
    import httpx

__all__ = ["httpx"]
