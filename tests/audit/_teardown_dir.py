"""One audit directory and the run that fills it, over the teardown world.

``run`` stands in for the orchestrator: it creates what a run creates in the order the SDK creates
it -- project, then ``create_audit``, then every resource named with the audit id -- and records it
in the state the way the steps do. It imports nothing that only the teardown protocol has, so the
same scenarios can be driven against an older client. The steps themselves (the id they send, the
halt on a failed ``create_audit``) are tested against the real orchestrator elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from tests.audit._teardown_client import WorldClient
from tests.audit._teardown_world import CHAIN, Audit, Res, TeardownPlatform, World

from dagnam.audit.candidates import CandidateKind
from dagnam.audit.state import AuditState, StepState, load_state, save_state
from dagnam.audit.workspace import write_workload

HEAD = CandidateKind.HEAD_TUNE
DELETED = {"reason": "deleted"}


class Directory:
    """The audit directory and what a run leaves in it."""

    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = audit_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.world = World()
        self.platform = TeardownPlatform(self.world)
        self.client = WorldClient(self.platform)
        self.exits: list[tuple[str, int]] = []
        self.published_at: int | None = None
        """How many destructive acts the world had logged when the audit was created."""
        self.pre_publish: set[str] = set()
        """What this directory had created when its audit was created: no audit could have tagged it."""
        self.checked: list[tuple[str, str, str]] = []
        """The destructive acts a test has already judged."""
        self.answered = True
        """Whether the platform answered the last ``cancel`` or ``delete``."""
        self._n = 0
        save_state(audit_dir, AuditState())

    @property
    def state(self) -> AuditState:
        return load_state(self.audit_dir)

    def _new(self, kind: str) -> str:
        self._n += 1
        return f"{kind}-{self._n}"

    def run(
        self, *, workloads: int = 1, create_audit_fails: bool = False, publish: bool = True
    ) -> int:
        """One run: project, audit (first), then per workload dataset, run, version and endpoint."""
        state, p = self.state, self.platform
        if state.halted == DELETED:
            return self._exit("run", 1)
        if state.project_id is None:
            state.project_id = p.create(Res("project", self._new("proj"), chain=CHAIN), None).id
        if publish and state.audit_id is None:
            if create_audit_fails:
                state.halted = {"reason": "publish_failed"}
                save_state(self.audit_dir, state)
                return self._exit("run", 1)
            state.audit_id = p.create_audit(Audit(self._new("audit"), project=state.project_id)).id
            self.published_at = len(self.world.destroyed)
            self.pre_publish = {r.id for r in self.world.res.values() if r.chain == CHAIN}
            state.tagged, state.claim_pending = True, True
            save_state(self.audit_dir, state)
        if state.audit_id is not None and state.claim_pending and not self._claim(state):
            return self._exit("run", 1)  # the claim failed as a whole: the run halts, asks again
        audit = self.world.audits.get(state.audit_id or "")
        if audit is not None:
            if audit.status == "deleted":
                state.halted = dict(DELETED)
                save_state(self.audit_dir, state)
                return self._exit("run", 1)
            audit.status = "live"
        state.halted = None
        if audit is not None:
            for step in state.all_steps():  # what the steps name, the platform records
                for kind, rid in (
                    ("dataset", step.dataset_id),
                    ("training_job", step.training_job_id),
                    ("deployment", step.deployment_id),
                ):
                    if rid is not None:
                        p.record(audit, kind, rid)
        for i in range(workloads):
            self._workload(state, f"w{i + 1}")
        save_state(self.audit_dir, state)
        return self._exit("run", 0)

    def _workload(self, state: AuditState, wid: str) -> None:
        if HEAD in state.workloads.get(wid, {}):
            return
        write_workload(
            self.audit_dir, wid, [{"input": "a", "label": "x"}], {"train": [0]}, {"format_key": "x"}
        )
        p, tag = self.platform, state.audit_id
        audit = self.world.audits.get(tag or "")
        ds = p.create(Res("dataset", self._new("ds"), chain=CHAIN), tag)
        ver = p.create(
            Res("model_version", self._new("ver"), project=state.project_id, chain=CHAIN), tag
        )
        job = p.create(
            Res(
                "training_job",
                self._new("job"),
                project=state.project_id,
                chain=CHAIN,
                dataset=ds.id,
                version=ver.id,
            ),
            tag,
        )
        dep = p.create(
            Res("deployment", self._new("dep"), project=state.project_id, chain=CHAIN), tag
        )
        ver.served_by.add(dep.id)
        if audit is not None:
            for kind, res in (("dataset", ds), ("training_job", job), ("deployment", dep)):
                p.record(audit, kind, res.id)
        state.workloads[wid] = {
            HEAD: StepState(
                dataset_id=ds.id,
                training_job_id=job.id,
                model_version_id=ver.id,
                deployment_id=dep.id,
                run_status="completed",
                deploy_status="running",
            )
        }

    def _claim(self, state: AuditState) -> bool:
        """Hand the earlier run's ids to the audit (nothing, in the base); whether it went through."""
        state.claim_pending = False
        save_state(self.audit_dir, state)
        return True

    def _exit(self, verb: str, code: int) -> int:
        self.exits.append((verb, code))
        return code
