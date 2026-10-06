#!/usr/bin/env python3
"""Release gate: refuse to publish an SDK its platform cannot serve yet.

The SDK redacts with the ``dagnam-contracts`` it installs and the platform
re-scans with its own, so the contract's class list is a wire protocol between
the two. An SDK released ahead of its platform uploads the rows and then stops
every audit at the PII check. This asks the platform which contract it runs
(``GET /health/build``, the ``contracts`` key) and exits non-zero unless its
``(major, minor)`` is at least the floor this repository declares in
``pyproject.toml`` (``dagnam-contracts>=X.Y``).

Anything short of a clear yes fails: a platform behind the floor, one too old
to report the key, and one that cannot be reached all leave the release
unproven. Deploy the platform first, then tag.

Standard library only: the release workflow runs it on a bare runner, before
anything is installed.

Usage:
    python scripts/check_platform.py https://api.dagnam.ai
"""

from __future__ import annotations

import argparse
from http.client import HTTPMessage
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import IO, override
import urllib.error
import urllib.parse
import urllib.request

CONTRACT = "dagnam-contracts"
BUILD_PATH = "/health/build"
TIMEOUT_SECONDS = 15.0
USER_AGENT = "dagnam-release-gate"
"""Named, because bot filters in front of an API commonly refuse the default ``Python-urllib``."""
_NAMED = re.compile(rf"^{CONTRACT}(?![\w.-])")
"""The requirement is for this distribution, not for one whose name merely starts with it."""
_FLOOR = re.compile(r">=\s*(\d+)\.(\d+)")
_MINOR = re.compile(r"(\d+)\.(\d+)")


def contract_floor(pyproject: Path) -> tuple[int, int]:
    """``(major, minor)`` of the ``dagnam-contracts>=X.Y`` this project declares."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    for requirement in project["dependencies"]:
        named = _NAMED.match(requirement)
        found = _FLOOR.search(requirement[named.end() :]) if named else None
        if found:
            return int(found[1]), int(found[2])
    sys.exit(f"{pyproject} declares no {CONTRACT}>=X.Y floor")


class _SameHostOnly(urllib.request.HTTPRedirectHandler):
    """Follow a redirect on the platform's own host and refuse one anywhere else.

    A gate that reads a version off whatever a redirect leads to proves nothing about the
    platform it was pointed at.
    """

    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            raise urllib.error.HTTPError(
                req.full_url, code, f"redirected to another host ({newurl})", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameHostOnly)


def platform_contract(api_url: str, timeout: float = TIMEOUT_SECONDS) -> object:
    """The platform's ``contracts`` value, as it arrived (``None`` when the key is absent)."""
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{BUILD_PATH}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with _OPENER.open(request, timeout=timeout) as response:
        build = json.load(response)
    return build.get("contracts") if isinstance(build, dict) else None


def main(argv: list[str] | None = None) -> int:
    """Exit status: 0 when the platform runs the declared contract floor or newer, else 1."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("api_url", help="The platform's API base URL.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
        help="The project file that declares the contract floor.",
    )
    args = parser.parse_args(argv)
    floor = contract_floor(args.pyproject)
    needs = f"this release needs {floor[0]}.{floor[1]} or newer. Deploy the platform first."
    try:
        contracts = platform_contract(args.api_url, TIMEOUT_SECONDS)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{args.api_url}{BUILD_PATH} could not be read ({exc}); {needs}\n")
        return 1
    found = _MINOR.match(contracts) if isinstance(contracts, str) else None
    if found is None:
        sys.stderr.write(f"{args.api_url} does not report its {CONTRACT} version; {needs}\n")
        return 1
    if (int(found[1]), int(found[2])) < floor:
        sys.stderr.write(f"{args.api_url} runs {CONTRACT} {contracts}; {needs}\n")
        return 1
    sys.stdout.write(f"{args.api_url} runs {CONTRACT} {contracts}: at or above the floor\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
