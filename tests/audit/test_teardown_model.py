"""Every teardown action sequence over eight scenarios keeps the protocol's invariants.

The scenarios are a deployment the owner moved, a dataset they linked elsewhere, a step that names
someone else's dataset, a run whose ``create_audit`` failed first, two finished workloads, a slice
that refuses once, a ``--local-only`` audit with the owner's own work beside it (published later),
and one whose dataset the owner linked elsewhere before it was published (the claim refuses it).
Each is driven through every sequence of up to three of the eight actions (delete; delete when the
platform does not answer, answers 404 for this key, or its walk dies after one step; cancel; cancel
when it does not answer or answers 404; run; run while the claim request fails) over the SDK's real
``cancel`` and ``delete`` and a platform that answers as the contract's table says. After every step:

* for a published audit the SDK touched only what the platform refused to claim, which this
  directory created;
* nothing the directory did not create is destroyed, by anyone, and no registry entry by the SDK;
* a cancel never marks a finished run cancelled;
* a delete exits 0 only if nothing of the audit's own is left and the audit record is gone, and
  exits 1 only if something is, or the platform did not answer, or its audit record is still there;
* a 404 never marks the directory deleted and never removes a local row.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from pathlib import Path

import pytest
from tests.audit._teardown_sdk import Sdk
from tests.audit._teardown_world import CHAIN, SDK, Res

from dagnam.audit.state import DELETED_STATE

ACTIONS = (
    "delete",
    "delete!500",
    "delete!404",
    "delete!crash1",
    "cancel",
    "cancel!500",
    "cancel!404",
    "run",
    "run!claim500",
)
Scenario = Callable[[Sdk], None]


def _ids(sdk: Sdk) -> tuple[str, str, str, str]:
    step = next(iter(sdk.state.workloads["w1"].values()))
    ids = (step.dataset_id, step.training_job_id, step.model_version_id, step.deployment_id)
    found = tuple(i for i in ids if i is not None)
    assert len(found) == len(ids)
    return found[0], found[1], found[2], found[3]


def moved_deployment(sdk: Sdk) -> None:
    """The owner moved the audit's endpoint to another project before anything else."""
    sdk.run()
    sdk.world.add(Res("project", "Q"))
    sdk.world.res[_ids(sdk)[3]].project = "Q"


def linked_elsewhere(sdk: Sdk) -> None:
    """The owner linked the audit's dataset to another project."""
    sdk.run()
    sdk.world.add(Res("project", "Q"))
    sdk.world.res[_ids(sdk)[0]].links.add("Q")


def named_foreign(sdk: Sdk) -> None:
    """A step names a dataset the owner already had: naming is never owning."""
    sdk.world.add(Res("project", "Q"))
    sdk.world.add(Res("dataset", "D-users", links={"Q"}))
    sdk.run()
    sdk.platform.record(sdk.world.audits[sdk.state.audit_id or ""], "dataset", "D-users")


def late_audit(sdk: Sdk) -> None:
    """The first run's ``create_audit`` failed (nothing uploaded); the second one made everything."""
    assert sdk.run(create_audit_fails=True) == 1
    sdk.run()


def two_runs(sdk: Sdk) -> None:
    """Two finished workloads."""
    sdk.run(workloads=2)


def refused_run(sdk: Sdk) -> None:
    """The owning slice refuses the run's delete (for one delete only)."""
    sdk.run()
    sdk.platform.refuse.add(_ids(sdk)[1])


def local_only(sdk: Sdk) -> None:
    """No platform record: the owner's own project mate, weights and endpoint sit beside ours."""
    sdk.run(publish=False)
    proj = sdk.state.project_id
    sdk.world.add(Res("dataset", "D-foreign", links={proj or ""}))
    sdk.world.add(Res("model_version", "ver-foreign", project=proj))
    sdk.world.add(Res("deployment", "dep-foreign", project="Q"))


def claim_refused(sdk: Sdk) -> None:
    """The owner linked the earlier local-only run's dataset elsewhere: the claim refuses it."""
    sdk.run(publish=False)
    sdk.world.add(Res("project", "Q"))
    sdk.world.res[_ids(sdk)[0]].links.add("Q")


SCENARIOS: tuple[Scenario, ...] = (
    moved_deployment,
    linked_elsewhere,
    named_foreign,
    late_audit,
    two_runs,
    refused_run,
    local_only,
    claim_refused,
)


def _apply(sdk: Sdk, action: str) -> None:
    verb, _, failure = action.partition("!")
    state = sdk.state
    was_deleted = state.halted == DELETED_STATE
    had_rows = (sdk.audit_dir / "workloads").exists()
    if verb == "delete":
        sdk.delete(failure or None)
        sdk.platform.refuse.clear()  # a refusal is transient: the next walk finds the slice willing
    elif verb == "cancel":
        sdk.cancel(failure or None)
    else:
        sdk.client.claim_failure = failure == "claim500"
        sdk.run()
    if failure == "404" and not was_deleted and state.audit_id is not None:
        assert sdk.state.halted != DELETED_STATE, "a 404 marked the directory deleted"
        assert (sdk.audit_dir / "workloads").exists() == had_rows, "a 404 removed local rows"


def _own_left(sdk: Sdk) -> list[str]:
    """What the directory created that is still on the platform and was not decided to be kept."""
    state = sdk.state
    audit = sdk.world.audits.get(state.audit_id or "")
    left: list[str] = []
    for r in sdk.world.res.values():
        if r.chain != CHAIN or r.kind == "project":
            continue
        present = r.bytes if r.kind == "model_version" else r.alive
        served_elsewhere = audit is not None and r.kind == "model_version" and bool(r.served_by)
        moved = (r.kind == "dataset" and bool(r.links - {state.project_id or ""})) or (
            r.kind in ("training_job", "deployment") and r.project != state.project_id
        )
        if present and not served_elsewhere and not moved:
            left.append(f"{r.kind} {r.id}")
    return left


def _untagged_before(sdk: Sdk, rid: str) -> bool:
    """Whether ``rid`` was made before the audit existed and the platform never tagged it."""
    return rid in sdk.pre_publish and sdk.world.res[rid].tag is None


def violations(sdk: Sdk, action: str) -> list[str]:
    """The invariants broken after ``action``."""
    out: list[str] = []
    state, world = sdk.state, sdk.world
    created = {r.id for r in world.res.values() if r.chain == CHAIN}
    fresh = len(sdk.checked)
    sdk.checked.extend(world.destroyed[fresh:])
    for index, (actor, what, rid) in enumerate(world.destroyed):
        # Judged on the tags as they stood when the act was made: a later claim tags it, not undoes it.
        after_publish = sdk.published_at is not None and index >= sdk.published_at
        if index >= fresh and actor == SDK and after_publish and not _untagged_before(sdk, rid):
            out.append(f"the SDK touched {what} {rid} of a published audit")
        if index >= fresh and rid not in created and what not in ("stop", "pause"):
            out.append(f"{what} {rid} destroyed by {actor}, and the directory did not create it")
        if index >= fresh and actor == SDK and what == "model_entry":
            out.append(f"the SDK deleted the registry entry of {rid}")
    for steps in state.workloads.values():
        for step in steps.values():
            job = world.res.get(step.training_job_id or "")
            if job is not None and job.status == "completed" and step.run_status == "cancelled":
                out.append(f"finished run {job.id} marked cancelled")
    if action.startswith("delete"):
        code = sdk.exits[-1][1]
        left = _own_left(sdk)
        audit = world.audits.get(state.audit_id or "")
        record_left = audit is not None and audit.status != "deleted"
        if code == 0 and (left or record_left):
            out.append(f"delete exit 0 with {left or 'the audit record'} left")
        if code == 1 and sdk.answered and not left and not record_left:
            out.append("delete exit 1 with nothing left and the platform answering")
    return out


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_every_sequence_keeps_the_invariants(scenario: Scenario, tmp_path: Path) -> None:
    found: list[str] = []
    for n in range(1, 4):
        for sequence in product(ACTIONS, repeat=n):
            sdk = Sdk(tmp_path / f"{len(found)}-{'-'.join(sequence)}")
            scenario(sdk)
            for step, action in enumerate(sequence):
                _apply(sdk, action)
                found += [f"{sequence[: step + 1]}: {v}" for v in violations(sdk, action)]
    assert not found, "\n".join(found[:8])
