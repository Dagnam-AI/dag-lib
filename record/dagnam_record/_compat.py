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
MAX_VERSION = 64
"""The longest version text that is believed: a longer one is no version of any release."""


def _release(text: str) -> Optional[Tuple[int, int, int]]:
    """The ``(major, minor, patch)`` at the start of ``text``, or ``None`` if it has none."""
    found = _RELEASE.match(text)
    if found is None:
        return None
    major, minor, patch = found.groups()
    return int(major), int(minor), int(patch or 0)


def in_range(version: str, bounds: Tuple[str, str]) -> bool:
    """Whether ``version`` satisfies the lower (``>=``) and upper (``<``) bound. Never raises.

    A pre-release counts as its release (``3.0.0rc1`` is 3.0.0); text with no release, or that
    cannot be read (digits past Python's integer limit), is out.
    """
    try:
        found = _release(version)
        low, high = _release(bounds[0][2:] + ".0"), _release(bounds[1][1:] + ".0")
        return found is not None and low is not None and high is not None and low <= found < high
    except Exception:  # ``int`` refuses a run of thousands of digits
        return False


def version_of(name: str) -> Optional[str]:
    """The ``__version__`` of the imported SDK ``name``; ``None`` if it is not imported. Never raises.

    ``UNKNOWN`` when it has none, when it is not text or is longer than ``MAX_VERSION``, or when
    reading it fails (a module with a lazy ``__getattr__`` that raises).
    """
    module = sys.modules.get(name)
    if module is None:
        return None
    try:
        version = getattr(module, "__version__", None)
    except Exception:  # a broken SDK must not break the program that imports the recorder
        return UNKNOWN
    if isinstance(version, str) and 0 < len(version) <= MAX_VERSION:
        return version
    return UNKNOWN


def is_untested(name: str) -> bool:
    """Whether the imported SDK ``name`` is outside the range the recorder was tested with.

    False when ``name`` has no range or is not imported (nothing of it can be patched). Read from
    the SDK as it is imported now, not from a flag set by ``install``, so a call is never recorded
    raw in the moment between the patch going live and the version being noted.
    """
    bounds = RANGES.get(name)
    version = version_of(name)
    return bounds is not None and version is not None and not in_range(version, bounds)
