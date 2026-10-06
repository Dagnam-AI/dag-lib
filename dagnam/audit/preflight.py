"""Ask the platform which privacy contract it runs, before anything is uploaded.

The SDK redacts rows with the ``dagnam-contracts`` it installs and the platform
re-scans them with its own, so the contract's class list is a wire protocol
between the two. An SDK ahead of its platform uploads the rows and then stops
every workload at the PII check, so :func:`check_platform` reads the
platform's version once, first:

* the platform's ``(major, minor)`` is behind this install's, or it does not
  say (no ``contracts`` key, no route, a body that is not a JSON object): a
  refusal, before any upload or credit is spent;
* the same ``(major, minor)`` with a newer patch: a warning naming the upgrade,
  because a patch changes what is found inside existing classes;
* equal, an older patch, or a newer minor or major: nothing to say here (a
  newer minor that matters stops the PII check, which says so);
* the request itself failing (the network, a 5xx): nothing proven, the run
  goes on and its own calls report the outage.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import re

from dagnam._core.exceptions import APIError, DagnamError, ResponseError
from dagnam.audit.steps import PlatformClient

CONTRACT = "dagnam-contracts"
"""The distribution both sides install: its PII classes and score shape are their wire protocol."""
PLATFORM_TOO_OLD = "platform_too_old"
"""Why a run stops before its first upload: the platform's contract is behind this SDK's."""
NOT_FOUND = 404
UPGRADE = f"pip install -U {CONTRACT}"
_RELEASE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

type Release = tuple[int, int, int]


class PlatformTooOldError(DagnamError):
    """The platform runs an older ``dagnam-contracts`` than this SDK, or does not say."""


@dataclass(frozen=True, slots=True)
class PlatformCheck:
    """What :func:`check_platform` found out."""

    contracts: str | None
    """The version the platform reported; ``None`` when it did not say or could not be asked."""
    refusal: str | None = None
    """Why this run must not start, or ``None``."""
    warning: str | None = None
    """What to say before the run starts, or ``None``."""


def installed_contract() -> str:
    """The ``dagnam-contracts`` version this install redacts and scores with."""
    return version(CONTRACT)


def release(text: object) -> Release | None:
    """``(major, minor, patch)`` of a version string; ``None`` for anything that does not start with one."""
    found = _RELEASE.match(text) if isinstance(text, str) else None
    return None if found is None else (int(found[1]), int(found[2]), int(found[3] or 0))


def version_gap(theirs: str | None, ours: str) -> str:
    """One sentence on how the platform's contract differs from this install's, or ``""``.

    Said when the two differ at any level; what to do depends on which is newer.
    """
    platform, installed = release(theirs), release(ours)
    if platform is None or installed is None or platform == installed:
        return ""
    if platform > installed:
        return (
            f" The platform runs {CONTRACT} {theirs}, this install {ours}: `{UPGRADE}`,"
            " then scan again."
        )
    return f" The platform runs {CONTRACT} {theirs}, older than this install's {ours}: wait for it."


def check_platform(client: PlatformClient) -> PlatformCheck:
    """The platform's contract, judged against the one this SDK installed."""
    ours = installed_contract()
    try:
        reported = client.get_platform_build().get("contracts")
    except ResponseError:
        reported = None  # answered, but not with a build document: it cannot say
    except APIError as exc:
        if exc.status_code != NOT_FOUND:
            return PlatformCheck(contracts=None)
        reported = None
    running, installed = release(reported), release(ours) or (0, 0, 0)
    if running is None or running[:2] < installed[:2]:
        said = (
            f"does not report its {CONTRACT} version"
            if running is None
            else f"runs {CONTRACT} {reported}"
        )
        return PlatformCheck(
            contracts=None if running is None else str(reported),
            refusal=(
                f"{PLATFORM_TOO_OLD}: the platform at {client.api_url} {said} and this SDK runs"
                f" {ours}. The platform has not been upgraded for this SDK yet; nothing was"
                " uploaded or spent by this run. Run again once it is."
            ),
        )
    warning = (
        f"the platform runs {CONTRACT} {reported} and this install {ours}: a newer patch can find"
        f" more inside the same classes, so rows redacted here may stop at the platform's PII"
        f" check. Run `{UPGRADE}` and scan again to avoid it."
        if running > installed and running[:2] == installed[:2]
        else None
    )
    return PlatformCheck(contracts=str(reported), warning=warning)


__all__ = [
    "CONTRACT",
    "PLATFORM_TOO_OLD",
    "UPGRADE",
    "PlatformCheck",
    "PlatformTooOldError",
    "check_platform",
    "installed_contract",
    "release",
    "version_gap",
]
