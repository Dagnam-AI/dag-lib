"""The platform's replay cache for a create route, in memory, for the resumable-create tests."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from dagnam._core.client import DagnamClient
from dagnam._types import JsonObject

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
RUNS = f"{API}/api/v1/training/foundation-runs"
RUN: JsonObject = {"project_id": "p1", "dataset_version_id": "ds-1-v2", "recipe_key": "r@1.2"}
LOST = "lost"
"""A scripted outcome: the platform commits the create and the client never hears the answer."""
IN_PROGRESS_REPLY: JsonObject = {"detail": "A request with this Idempotency-Key is in progress"}
TRANSIENT = frozenset({429, 500, 502, 503, 504})


class Replays:
    """The platform's replay cache for one create route, as its middleware keeps it."""

    def __init__(self, made: str = "run_id") -> None:
        self.made = made
        self.outcomes: list[int | str] = []
        """What the handler answers next, in order: a status, or :data:`LOST`; 201 once it runs dry."""
        self.created: list[JsonObject] = []
        """Bodies the handler really created something for: a replay never adds one."""
        self.keys: list[str | None] = []
        self.deleted: set[str] = set()
        """Ids of what the owner removed since: a read of one is the platform's 404."""
        self.read_status: int | None = None
        """What every read of a created resource answers instead (a 5xx), when set."""
        self.route = ""
        self.pointer_only = False
        """Replays answer with the platform's marker, no body: ``resource_id``, no ``id`` of the resource."""
        self.with_location = False
        """The marker also carries the original's ``Location`` (the platform sends it only when the original had one)."""
        self.marked = False
        """The "in progress" answer carries the machine-readable marker beside the text."""
        self.in_progress = 0
        """How many asks the platform answers 409 "in progress" before it lets one through."""
        self.reads: list[str] = []
        self._cache: dict[str, tuple[str, int, JsonObject]] = {}

    def read(self, request: Any, context: Any) -> JsonObject:
        """``GET <route>/<id>``: the resource, or the 404 of one that is gone."""
        item_id = request.path.rsplit("/", 1)[-1]
        self.reads.append(item_id)
        if self.read_status is not None:
            context.status_code = self.read_status
            return {"detail": "down"}
        if item_id in self.deleted:
            context.status_code = 404
            return {"detail": "not found"}
        return {self.made: item_id}

    def __call__(self, request: Any, context: Any) -> JsonObject:
        key = request.headers.get("Idempotency-Key")
        self.keys.append(key)
        if self.in_progress:
            self.in_progress -= 1
            context.status_code = 409
            return (
                {**IN_PROGRESS_REPLY, "error": "idempotency_in_progress"}
                if self.marked
                else IN_PROGRESS_REPLY
            )
        fingerprint = hashlib.sha256(request.text.encode()).hexdigest()
        if key in self._cache:
            cached_print, status, body = self._cache[key]
            if cached_print != fingerprint:
                context.status_code = 422
                return {"detail": "Idempotency-Key reused with different parameters"}
            context.status_code = status
            context.headers["Idempotency-Replayed"] = "true"
            if self.pointer_only and status == 201:
                return self._marker(context, status, str(body[self.made]))
            return body
        outcome = self.outcomes.pop(0) if self.outcomes else 201
        status = 201 if outcome == LOST else int(outcome)
        body: JsonObject = {"detail": f"refused with {status}"}
        if status == 201:
            self.created.append(request.json())
            body = {self.made: f"made-{len(self.created)}", "status": "queued", "api_key": None}
        if key is not None and status not in TRANSIENT:
            self._cache[key] = (fingerprint, status, body)
        if outcome == LOST:
            raise KeyboardInterrupt
        context.status_code = status
        return body

    def _marker(self, context: Any, status: int, resource_id: str) -> JsonObject:
        """The replay of a create whose body was not retained, exactly as the platform words it."""
        context.headers["Idempotency-Body-Retained"] = "false"
        if self.with_location:
            context.headers["Location"] = f"{self.route}/{resource_id}"
        return {
            "resource_id": resource_id,
            "status": status,
            "replayed_without_body": True,
            "detail": (
                f"This request already completed with status {status}; its response body was"
                " not retained. Read the resource by its id."
            ),
        }


def new_client(*, resume: bool = True) -> DagnamClient:
    """A fresh client, as each ``dagnam`` process builds one."""
    client = DagnamClient(API, "k")
    client._sleep = lambda _s: None
    client.resume_creates = resume
    return client


def mount(requests_mock: RequestsMocker, route: str, server: Replays) -> Replays:
    """The platform's create route and the read of what it made, for ``route``."""
    server.route = route
    requests_mock.post(route, json=server)
    for number in range(1, 7):  # the ids the fake mints: made-1 ...
        requests_mock.get(f"{route}/made-{number}", json=server.read)
    requests_mock.get(f"{route}/down", json=server.read)
    return server
