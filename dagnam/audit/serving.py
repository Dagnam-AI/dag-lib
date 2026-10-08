"""The all-or-nothing check before this client deletes an endpoint on its own.

The platform refuses to delete an audit while one of its endpoints is serving
(``409 endpoints_serving``). The ids this client deletes itself -- an unpublished audit's, and
those the platform refused to claim -- get the same rule: every one of them is read BEFORE the
first destructive call, and if any may be serving, nothing is deleted and
:class:`~dagnam._core.exceptions.EndpointsServingError` says which. A read cannot tell whether an
endpoint that is still rolling out ever served, so ``deploying`` counts as serving: the safe
answer is the one that deletes nothing. Not found counts as not serving; any other failure to
read propagates, because deleting on a guess is the one thing this check exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable

from dagnam._core.exceptions import DeploymentNotFoundError, EndpointsServingError
from dagnam._types import JsonObject
from dagnam.audit.cleanup_kinds import CleanupClient

MAY_SERVE = frozenset({"running", "deploying"})
"""The deployment statuses that block a delete: answering now, or on its way to answering."""


def serving_here(client: CleanupClient, ids: Iterable[str]) -> list[JsonObject]:
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
        status = row.get("status")
        if status in MAY_SERVE:
            name = row.get("name")
            found.append(
                {
                    "id": deployment_id,
                    "name": name if isinstance(name, str) and name else deployment_id,
                    "status": status,
                }
            )
    return found


def refuse_if_serving(client: CleanupClient, ids: Iterable[str], *, include: bool) -> None:
    """Raise :class:`EndpointsServingError` if any of ``ids`` may be serving; ``include`` skips the check."""
    if include:
        return
    found = serving_here(client, ids)
    if found:
        one = len(found) == 1
        raise EndpointsServingError(
            f"Nothing was deleted: {len(found)} {'endpoint' if one else 'endpoints'} of this audit "
            f"may still be serving, and deleting {'it' if one else 'them'} would break any app "
            f"that calls {'it' if one else 'them'}. Stop {'it' if one else 'them'} first "
            "(`dagnam audit cancel`), or delete "
            f"{'it' if one else 'them'} anyway with include_endpoints=True.",
            found,
        )
