"""The SDK versions the recorder was tested with.

Outside a range the recorder still records, but writes the provider-neutral ``norm`` only (never
the raw request and response, whose shape a new SDK may have changed) and says so once. The
version is read from the imported SDK module (``__version__``), not from package metadata, so the
recorder imports nothing for it.
"""

from __future__ import annotations

import re
import sys
from typing import Optional, Tuple

OPENAI = (">=1.66", "<3")
ANTHROPIC = (">=0.40", "<1")
RANGES = {"openai": OPENAI, "anthropic": ANTHROPIC}
UNKNOWN = "unknown"
_RELEASE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def _release(text: str) -> Optional[Tuple[int, int, int]]:
    """The ``(major, minor, patch)`` at the start of ``text``, or ``None`` if it has none."""
    found = _RELEASE.match(text)
    if found is None:
        return None
    major, minor, patch = found.groups()
    return int(major), int(minor), int(patch or 0)


def in_range(version: str, bounds: Tuple[str, str]) -> bool:
    """Whether ``version`` satisfies the lower (``>=``) and upper (``<``) bound.

    A pre-release counts as its release (``3.0.0rc1`` is 3.0.0); text with no release is out.
    """
    found = _release(version)
    low, high = _release(bounds[0][2:] + ".0"), _release(bounds[1][1:] + ".0")
    return found is not None and low is not None and high is not None and low <= found < high


def version_of(name: str) -> Optional[str]:
    """The ``__version__`` of the imported SDK ``name``; ``UNKNOWN`` if it has none; ``None`` if absent."""
    module = sys.modules.get(name)
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) and version else UNKNOWN
