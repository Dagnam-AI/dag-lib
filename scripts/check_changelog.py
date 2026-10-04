#!/usr/bin/env python3
"""Release gate: refuse to publish a version whose changelog entries are in the wrong place.

At release time every entry is folded out of ``## [Unreleased]`` into a dated
``## [X.Y.Z] - YYYY-MM-DD`` heading, and ``X.Y.Z`` is the ``__version__`` being
published. This exits non-zero when ``[Unreleased]`` still holds anything, or
when the newest dated heading is not the current version with a real date: the
two ways a release ships its changes undocumented, or under a date it does not
have yet.

Standard library only.

Usage:
    python scripts/check_changelog.py
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
_HEADING = re.compile(r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\S+))?\s*$", re.MULTILINE)
_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


def problems(changelog: str, version: str) -> list[str]:
    """What stops ``changelog`` from being the one ``version`` is published with."""
    headings = list(_HEADING.finditer(changelog))
    found: list[str] = []
    unreleased = next((h for h in headings if h["name"] == "Unreleased"), None)
    if unreleased is None:
        found.append("there is no `## [Unreleased]` heading to leave empty for the next cycle")
    else:
        following = next((h for h in headings if h.start() > unreleased.start()), None)
        body = changelog[unreleased.end() : following.start() if following else len(changelog)]
        if body.strip():
            found.append(
                "`## [Unreleased]` still holds entries: fold them into"
                f" `## [{version}] - YYYY-MM-DD` and leave it empty"
            )
    newest = next((h for h in headings if h["name"] != "Unreleased"), None)
    if newest is None or newest["name"] != version:
        found.append(f"the newest dated heading is not `## [{version}] - YYYY-MM-DD`")
    elif not _is_date(newest["date"]):
        found.append(f"`## [{version}]` has no real release date ({newest['date']!r})")
    return found


def _is_date(value: str | None) -> bool:
    try:
        date.fromisoformat(value or "")
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Exit status: 0 when the changelog is ready to release, else 1 with every reason on stderr."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--changelog", type=Path, default=ROOT / "CHANGELOG.md")
    parser.add_argument("--init", type=Path, default=ROOT / "dagnam" / "__init__.py")
    args = parser.parse_args(argv)
    declared = _VERSION.search(args.init.read_text(encoding="utf-8"))
    if declared is None:
        sys.stderr.write(f"{args.init} declares no __version__\n")
        return 1
    found = problems(args.changelog.read_text(encoding="utf-8"), declared[1])
    for reason in found:
        sys.stderr.write(f"{args.changelog}: {reason}\n")
    if found:
        return 1
    sys.stdout.write(f"{args.changelog} is ready for {declared[1]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
