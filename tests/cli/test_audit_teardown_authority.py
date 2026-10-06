"""``audit delete`` and ``audit cancel`` against a platform that tags what an audit creates.

Each test is one way an SDK that infers what the platform did, and then "finishes the job" with
calls of its own, went wrong: it destroyed what the platform had kept, finished a refused delete
from here and exited 0 with the audit still halted, marked a finished run cancelled, and deleted a
registry entry around weights it had no say over after the platform did not answer. Run through the
command line, over a real ``state.json``, against the designed platform: the platform's two audit
routes are the only paths that touch a published audit, and a destructive call of the client's own
is what every test here rules out.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest
from tests.audit._teardown_client import WorldClient
from tests.audit._teardown_dir import HEAD, Directory
from tests.audit._teardown_world import SDK, Res

from dagnam.audit.state import load_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


class Audit:
    """A published audit on disk, the world it lives in, and the two commands."""

    def __init__(
        self, run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, workloads: int = 1
    ) -> None:
        self.run_cli = run_cli
        self.directory = Directory(tmp_path / "audit")
        self.directory.run(workloads=workloads)
        self.client: WorldClient = self.directory.client
        monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: self.client)
        monkeypatch.setenv("DAGNAM_API_KEY", "dk-secret-0000")

    @property
    def world(self) -> object:
        return self.directory.world

    def _command(self, *argv: str) -> int:
        try:
            return self.run_cli(["audit", *argv, str(self.directory.audit_dir)])
        except SystemExit as exc:
            return int(str(exc.code))

    def delete(self, failure: str | None = None) -> int:
        self.client.audit_failure = failure
        return self._command("delete", "--yes")

    def cancel(self) -> int:
        return self._command("cancel")

    def sdk_acts(self) -> list[tuple[str, str, str]]:
        """Every destructive act the client itself made, whatever the platform said."""
        return [act for act in self.directory.world.destroyed if act[0] == SDK]

    def ids(self) -> tuple[str, str, str, str]:
        step = load_state(self.directory.audit_dir).workloads["w1"][HEAD]
        found = (step.dataset_id, step.training_job_id, step.model_version_id, step.deployment_id)
        assert None not in found
        return (str(found[0]), str(found[1]), str(found[2]), str(found[3]))


@pytest.fixture
def audit(run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> Audit:
    return Audit(run_cli, tmp_path, monkeypatch)


def test_a_second_delete_destroys_nothing_the_first_one_kept(audit: Audit) -> None:
    """The owner linked the dataset to another project; the platform kept it (``not_in_project``)."""
    dataset = audit.ids()[0]
    audit.directory.world.add(Res("project", "Q"))
    audit.directory.world.res[dataset].links.add("Q")

    assert audit.delete() == 0
    assert audit.delete() == 0

    assert audit.sdk_acts() == []
    assert audit.directory.world.res[dataset].alive  # the owner's, still there


def test_a_repeat_delete_the_platform_answers_with_a_404_destroys_nothing_either(
    audit: Audit,
) -> None:
    """An older platform, or the website, has already deleted the audit: a 404 decides nothing."""
    dataset = audit.ids()[0]
    audit.directory.world.add(Res("project", "Q"))
    audit.directory.world.res[dataset].links.add("Q")
    audit.directory.platform.tombstone = False

    assert audit.delete() == 0
    (audit.directory.audit_dir / "workloads").mkdir(exist_ok=True)
    assert audit.delete() == 0  # the 404: deleted elsewhere, nothing remote is touched

    assert audit.sdk_acts() == []
    assert audit.directory.world.res[dataset].alive


def test_a_cancel_after_the_delete_pauses_no_endpoint_the_platform_kept(audit: Audit) -> None:
    """The owner moved the endpoint to another project; the delete kept it. A cancel is then refused."""
    endpoint = audit.ids()[3]
    audit.directory.world.add(Res("project", "Q"))
    audit.directory.world.res[endpoint].project = "Q"

    assert audit.delete() == 0
    assert audit.cancel() == 1  # a deleted audit has nothing to cancel

    assert audit.sdk_acts() == []
    assert not audit.directory.world.res[endpoint].paused


def test_a_refused_delete_is_not_finished_from_here_and_the_exit_says_so(audit: Audit) -> None:
    job = audit.ids()[1]
    audit.directory.platform.refuse.add(job)

    assert audit.delete() == 1

    assert audit.sdk_acts() == []
    assert audit.directory.world.res[job].alive  # the platform's decision, not overridden
    assert load_state(audit.directory.audit_dir).halted is None  # nothing local was removed
    assert (audit.directory.audit_dir / "workloads").exists()

    audit.directory.platform.refuse.clear()  # the next delete asks the platform again
    assert audit.delete() == 0
    assert not audit.directory.world.res[job].alive


def test_a_cancel_never_marks_a_run_that_had_already_finished(audit: Audit) -> None:
    assert audit.cancel() == 0

    step = load_state(audit.directory.audit_dir).workloads["w1"][HEAD]
    assert step.run_status == "completed"  # the platform said ``already_stopped`` about it
    assert audit.sdk_acts() == []


def test_a_delete_the_platform_did_not_answer_leaves_every_resource_and_the_weights_alone(
    audit: Audit,
) -> None:
    dataset, job, version, endpoint = audit.ids()

    assert audit.delete(failure="500") == 1

    assert audit.sdk_acts() == []  # no entry delete, no direct delete: nothing was decided
    assert all(audit.directory.world.res[i].alive for i in (dataset, job, version, endpoint))
    assert audit.directory.world.res[version].bytes
    assert (audit.directory.audit_dir / "workloads").exists()

    assert audit.delete() == 0  # run again: the platform's own walk, by tag, removes all of it
    assert not audit.directory.world.res[version].bytes
    assert not any(what == "model_entry" for _, what, _ in audit.directory.world.destroyed)
