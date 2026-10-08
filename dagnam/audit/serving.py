"""The all-or-nothing check before this client deletes an endpoint on its own.

The platform refuses to delete an audit while one of its endpoints is serving
(``409 endpoints_serving``). The ids this client deletes itself -- an unpublished audit's, and
those the platform refused to claim -- get the same rule: every one of them is read BEFORE the
first destructive call, and if any may be serving, nothing is deleted and
:class:`~dagnam._core.exceptions.EndpointsServingError` says which.

The check fails closed. Only a status known to be quiet (:data:`QUIET`) lets an endpoint be
deleted; ``running``, ``deploying`` (a read cannot tell whether it ever served), a status a later
platform adds, and an answer with no usable status all count as serving. Not found counts as not
serving. Any other failure to read propagates, because deleting on a guess is the one thing this
check exists to prevent (a caller that must go on regardless asks for it to be counted as serving
instead: ``unreadable_is_serving``).
"""

from __future__ import annotations

from collections.abc import Iterable

from dagnam._core.exceptions import DagnamError, DeploymentNotFoundError, EndpointsServingError
from dagnam._types import JsonObject
from dagnam.audit.cleanup_kinds import CleanupClient

QUIET = frozenset({"paused", "stopped", "failed", "not_provisioned"})
"""The deployment statuses that answer no request and are not about to: the only ones deletable."""
UNREADABLE = "unreadable"
"""The ``status`` listed for an endpoint that could not be read when that is counted as serving."""


def serving_here(
    client: CleanupClient, ids: Iterable[str], *, unreadable_is_serving: bool = False
) -> list[JsonObject]:
    """The endpoints among ``ids`` that may be serving, as the platform's refusal lists them.

    Without ``last_request_at``: only the platform records requests, and a read of a deployment
    does not carry the newest one.
    """
    found: list[JsonObject] = []
    for deployment_id in ids:
        try:
            row = client.get_deployment(deployment_id)
        except DeploymentNotFoundError:
            continue
        except DagnamError:
            if not unreadable_is_serving:
                raise
            found.append({"id": deployment_id, "name": deployment_id, "status": UNREADABLE})
            continue
        status = row.get("status")
        if not isinstance(status, str) or status not in QUIET:
            name = row.get("name")
            found.append(
                {
                    "id": deployment_id,
                    "name": name if isinstance(name, str) and name else deployment_id,
                    "status": status if isinstance(status, str) else UNREADABLE,
                }
            )
    return found


def _sentence(found: list[JsonObject]) -> str:
    one = len(found) == 1
    it = "it" if one else "them"
    running = [e for e in found if e["status"] == "running"]
    advice = []
    if running:
        advice.append(
            f"Stop {'it' if len(running) == len(found) else 'the running ones'} first with `dagnam audit cancel`."
        )
    if len(running) < len(found):
        advice.append(
            "An endpoint that is still rolling out, or whose state is not known here, cannot be "
            "paused from here: wait for it to settle."
        )
    advice.append(f"Or delete {it} anyway with include_endpoints=True.")
    return (
        f"Nothing was deleted: {len(found)} {'endpoint' if one else 'endpoints'} of this audit "
        f"may still be serving, and deleting {it} would break any app that calls {it}. "
        + " ".join(advice)
    )


def refuse_if_serving(client: CleanupClient, ids: Iterable[str], *, include: bool) -> None:
    """Raise :class:`EndpointsServingError` if any of ``ids`` may be serving; ``include`` skips the check."""
    if include:
        return
    found = serving_here(client, ids)
    if found:
        raise EndpointsServingError(_sentence(found), found, from_platform=False)
