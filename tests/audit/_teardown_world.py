"""The resources a teardown acts on, and the designed platform that cancels and deletes them.

A small world of owner-scoped resources, each carrying the audit id the platform wrote at
creation (``tag``). :class:`TeardownPlatform` answers an audit's cancel and delete from that tag
alone, in the receipt shape of the designed contract (a ``code`` on every row, ``audit_status``,
a tombstone for a repeat delete, one cumulative row per ``(kind, id)``). Every destructive act is
logged with who did it, so a test can check a whole history, not only the end state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dagnam._core.exceptions import APIError
from dagnam._types import JsonObject, JsonValue

OWNER = "owner"
PLATFORM = "owner:platform"
SDK = "owner:sdk"
CHAIN = "chain-1"
"""The directory's process chain: what ``state.json`` can say it created."""
MISSING = 404


@dataclass
class Res:
    """One resource: its tag is the audit that was named when it was created, never later."""

    kind: str
    id: str
    project: str | None = None
    links: set[str] = field(default_factory=set)
    tag: str | None = None
    chain: str | None = None
    alive: bool = True
    status: str = "completed"
    dataset: str | None = None
    version: str | None = None
    bytes: bool = True
    revoked: bool = False
    served_by: set[str] = field(default_factory=set)
    paused: bool = False


@dataclass
class Audit:
    """The platform's record of one audit."""

    id: str
    project: str = "proj"
    status: str = "live"
    recorded: dict[str, list[str]] = field(default_factory=dict)
    receipt: dict[tuple[str, str], JsonObject] = field(default_factory=dict)
    attempts: int = 0


def row(kind: str, rid: str, status: str, code: str, reason: str | None = None) -> JsonObject:
    """A receipt row exactly as the contract words it."""
    return {"kind": kind, "id": rid, "status": status, "code": code, "reason": reason}


class World:
    """Every resource and every destructive act, in order."""

    def __init__(self) -> None:
        self.res: dict[str, Res] = {}
        self.audits: dict[str, Audit] = {}
        self.destroyed: list[tuple[str, str, str]] = []
        """``(actor, what, id)`` per destructive act: delete, purge, pause or stop."""

    def add(self, res: Res) -> Res:
        self.res[res.id] = res
        return res

    def get(self, rid: str) -> Res | None:
        found = self.res.get(rid)
        return found if found is not None and found.alive else None

    def alive(self, kind: str) -> list[Res]:
        return [r for r in self.res.values() if r.alive and r.kind == kind]

    def destroy(self, actor: str, rid: str) -> None:
        found = self.res[rid]
        found.alive = False
        found.bytes = False
        self.destroyed.append((actor, found.kind, rid))
        for other in self.res.values():
            other.served_by.discard(rid)

    def runs_naming(self, dataset_id: str) -> list[Res]:
        return [r for r in self.alive("training_job") if r.dataset == dataset_id]


class TeardownPlatform:
    """The designed platform's cancel and delete over a :class:`World`."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.refuse: set[str] = set()
        self.storage_fails = False
        self.tombstone = True
        """A deleted audit answers a repeat delete with its stored receipt; an older platform says 404."""

    def create_audit(self, audit: Audit, *, cli_project: bool = True) -> Audit:
        """Record the audit; a command-line audit tags the project it names if it is empty."""
        self.world.audits[audit.id] = audit
        proj = self.world.get(audit.project)
        if (
            cli_project
            and proj is not None
            and proj.tag is None
            and not self._holds_other(audit, proj)
        ):
            proj.tag = audit.id
        return audit

    def create(self, res: Res, audit_id: str | None) -> Res:
        """A create route: it writes the tag once, from the id the caller named."""
        if audit_id is not None:
            found = self.world.audits.get(audit_id)
            if found is None or found.status == "deleted":
                raise APIError(MISSING, "audit not found")
            res.tag = audit_id
        return self.world.add(res)

    def record(self, audit: Audit, kind: str, rid: str) -> None:
        ids = audit.recorded.setdefault(kind, [])
        if rid not in ids:
            ids.append(rid)

    # ------------------------------------------------------------------ entitlement
    def _holds_other(self, audit: Audit, proj: Res, own: set[str] | None = None) -> bool:
        own = own or set()
        for r in self.world.alive("training_job") + self.world.alive("deployment"):
            if r.project == proj.id and r.id not in own:
                return True
        for r in self.world.alive("dataset"):
            if proj.id in r.links and r.id not in own:
                return True
        for r in self.world.alive("model_version"):
            if r.project == proj.id and not r.revoked and r.id not in own:
                return True
        return any(
            o.project == proj.id and o.id != audit.id and o.status != "deleted"
            for o in self.world.audits.values()
        )

    def _own(self, audit: Audit) -> tuple[dict[str, list[str]], list[JsonObject]]:
        """The ids this audit acts on, by kind, and the rows it keeps: decided by the tag alone."""
        ids: dict[str, list[str]] = {
            "deployment": [],
            "training_job": [],
            "dataset": [],
            "model_version": [],
        }
        kept: list[JsonObject] = []
        for r in self.world.alive("deployment") + self.world.alive("training_job"):
            if r.tag == audit.id:
                if r.project != audit.project:
                    kept.append(row(r.kind, r.id, "kept", "not_in_project"))
                else:
                    ids[r.kind].append(r.id)
        for r in self.world.alive("dataset"):
            if r.tag == audit.id:
                if r.links - {audit.project}:
                    kept.append(row(r.kind, r.id, "kept", "not_in_project"))
                else:
                    ids[r.kind].append(r.id)
        ids["model_version"] = [
            r.id for r in self.world.alive("model_version") if r.tag == audit.id
        ]
        for kind in ("deployment", "training_job", "dataset"):
            for rid in audit.recorded.get(kind, []):
                r = self.world.get(rid)
                if r is not None and r.tag != audit.id:
                    kept.append(row(kind, rid, "kept", "not_created_here"))
        return ids, kept

    # ------------------------------------------------------------------ walks
    def _one(self, kind: str, rid: str, audit: Audit) -> JsonObject:
        w = self.world
        r = w.get(rid)
        if r is None:
            return row(kind, rid, "already_absent", "already_absent")
        if rid in self.refuse:
            return row(kind, rid, "blocked", "refused", "the owning slice refused")
        if kind == "training_job":
            w.destroy(PLATFORM, rid)
            return row(kind, rid, "deleted", "deleted")
        if kind == "dataset":
            if w.runs_naming(rid):
                return row(kind, rid, "blocked", "refused", "dataset in use")
            w.destroy(PLATFORM, rid)
            return row(kind, rid, "deleted", "deleted")
        if kind == "model_version":
            if r.served_by - {d.id for d in w.alive("deployment") if d.tag == audit.id}:
                return row(kind, rid, "kept", "weights_served")
            r.revoked = True
            if self.storage_fails:
                return row(kind, rid, "blocked", "weights_not_removed")
            if r.bytes:
                r.bytes = False
                w.destroyed.append((PLATFORM, "weights", rid))
            return row(kind, rid, "deleted", "deleted")
        w.destroy(PLATFORM, rid)
        return row(kind, rid, "deleted", "deleted")

    def _project_row(self, audit: Audit, ids: dict[str, list[str]], held: bool) -> JsonObject:
        proj = self.world.get(audit.project)
        own = {i for found in ids.values() for i in found}
        if proj is None or proj.tag != audit.id:
            return row("project", audit.project, "kept", "project_not_ours")
        if self._holds_other(audit, proj, own):
            return row("project", audit.project, "kept", "project_shared")
        if held:
            return row("project", audit.project, "kept", "project_held")
        self.world.destroy(PLATFORM, audit.project)
        return row("project", audit.project, "deleted", "deleted")

    @staticmethod
    def _merge(audit: Audit, new: JsonObject) -> None:
        """One cumulative row per id: ``deleted`` beats all, ``kept`` is final, ``blocked`` yields."""
        key = (str(new["kind"]), str(new["id"]))
        old = audit.receipt.get(key)
        keeps_old = (
            old is not None
            and old["status"] != "blocked"
            and new["status"] != "deleted"
            and (old["status"] == "kept" or new["status"] == "already_absent")
        )
        if not keeps_old:
            audit.receipt[key] = new

    def _receipt(self, audit: Audit, schema: str, entries: list[JsonObject]) -> JsonObject:
        rows: list[JsonValue] = list(entries)
        return {
            "schema": schema,
            "deleted_at": "2026-10-04T00:00:00Z",
            "audit_status": audit.status,
            "attempt": audit.attempts,
            "entries": rows,
        }

    def delete(self, audit_id: str, *, crash_at: int | None = None) -> JsonObject:
        """``DELETE /audits/{id}``; ``crash_at`` kills the request after that many steps (a 500)."""
        audit = self.world.audits.get(audit_id)
        if audit is None:
            raise APIError(MISSING, "audit not found")
        schema = "dagnam.audit.deleted/1"
        if audit.status == "deleted":
            if not self.tombstone:
                raise APIError(MISSING, "audit not found")
            return self._receipt(audit, schema, list(audit.receipt.values()))
        audit.status, audit.attempts = "halted", audit.attempts + 1
        ids, kept = self._own(audit)
        plan = [
            (k, i)
            for k in ("deployment", "model_version", "training_job", "dataset")
            for i in ids[k]
        ]
        rows: list[JsonObject] = []
        for n, (kind, rid) in enumerate(plan):
            if crash_at is not None and n == crash_at:
                raise APIError(500, "server error")
            rows.append(self._one(kind, rid, audit))
        rows += kept
        held = any(x["status"] == "blocked" or x["code"] == "weights_served" for x in rows)
        rows.append(self._project_row(audit, ids, held))
        for x in rows:
            self._merge(audit, x)
        if not any(x["status"] == "blocked" for x in rows):
            audit.status = "deleted"
        return self._receipt(audit, schema, list(audit.receipt.values()))

    def cancel(self, audit_id: str) -> JsonObject:
        """``POST /audits/{id}/cancel``: stop what is the audit's; a finished one is ``already_stopped``."""
        audit = self.world.audits.get(audit_id)
        if audit is None or audit.status == "deleted":
            raise APIError(MISSING, "audit not found")
        audit.status = "halted"
        w = self.world
        ids, kept = self._own(audit)
        rows: list[JsonObject] = []
        for rid in ids["deployment"]:
            r = w.res[rid]
            if r.paused:
                rows.append(row("deployment", rid, "stopped", "already_stopped"))
            else:
                r.paused = True
                w.destroyed.append((PLATFORM, "pause", rid))
                rows.append(row("deployment", rid, "stopped", "stopped"))
        for rid in ids["training_job"]:
            r = w.res[rid]
            if r.status != "running":
                rows.append(row("training_job", rid, "stopped", "already_stopped"))
            else:
                r.status = "cancelled"
                w.destroyed.append((PLATFORM, "stop", rid))
                rows.append(row("training_job", rid, "stopped", "stopped"))
        rows += [k for k in kept if k["kind"] != "dataset"]
        return {
            "schema": "dagnam.audit.cancelled/1",
            "deleted_at": "2026-10-04T00:00:00Z",
            "audit_status": "halted",
            "entries": [*rows],
        }
