"""``audit run``, ``cancel`` and ``delete`` never read or write through a link inside the audit directory."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest
from tests.audit._platform import FakePlatform
from tests.cli._audit_dirs import cli_state, platform_with_everything

from dagnam.audit.state import save_state

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="creating a symlink needs a privilege on Windows"
)


@pytest.fixture(autouse=True)
def no_real_keyring(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)


@pytest.mark.parametrize(
    "argv",
    [["audit", "cancel"], ["audit", "delete", "--yes"], ["audit", "run", "--yes"]],
    ids=["cancel", "delete", "run"],
)
def test_a_lock_file_that_is_a_link_ends_the_command_with_a_message_and_touches_nothing(
    run_cli: CliRunner,
    tmp_path: Path,
    capsys: StrCapture,
    monkeypatch: PytestMonkeyPatch,
    argv: list[str],
) -> None:
    fake = FakePlatform()
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    monkeypatch.setattr("dagnam.cli.audit_run.client_from_env", lambda: fake)
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    root = tmp_path / "audit"
    save_state(root, cli_state())
    (root / "state.json.lock").symlink_to(victim)

    with pytest.raises(SystemExit) as exc:
        run_cli([*argv[:2], str(root), *argv[2:]])

    assert exc.value.code == 1
    assert "symbolic link or leads outside the audit directory" in capsys.readouterr().err
    assert victim.read_text(encoding="utf-8") == "precious"
    assert fake.call_log == []


def test_a_workloads_folder_that_is_a_link_ends_a_run_before_anything_is_uploaded(
    run_cli: CliRunner,
    prepared_dir: Path,
    platform: FakePlatform,
    tmp_path: Path,
    capsys: StrCapture,
) -> None:
    moved = tmp_path / "moved"
    (prepared_dir / "workloads").rename(moved)
    (prepared_dir / "workloads").symlink_to(moved, target_is_directory=True)

    with pytest.raises(SystemExit) as exc:
        run_cli(["audit", "run", str(prepared_dir), "--yes"])

    assert exc.value.code == 1
    assert "symbolic link or leads outside the audit directory" in capsys.readouterr().err
    assert platform.call_log == []


@pytest.mark.parametrize("command", ["cancel", "delete"])
def test_an_audit_directory_named_through_a_link_works_and_is_resolved_once(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch, command: str
) -> None:
    """A temp or home directory is often a link: naming it is the user's choice, only links inside are refused."""
    fake = platform_with_everything()
    monkeypatch.setattr("dagnam.cli.audit_cleanup.client_from_env", lambda: fake)
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    real = tmp_path / "real"
    save_state(real, cli_state())
    named = tmp_path / "named"
    named.symlink_to(real, target_is_directory=True)

    argv = ["audit", command, str(named)]
    assert run_cli([*argv, "--yes"] if command == "delete" else argv) == 0

    assert (real / ("deleted.json" if command == "delete" else "cancelled.json")).exists()
