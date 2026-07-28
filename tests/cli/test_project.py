"""CLI projects subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


# ---------------------------------------------------------------- projects


def test_projects_list_get_create_delete_duplicate(run_cli: CliRunner, capsys: StrCapture) -> None:
    project_id = "4c857368-86ee-4fab-9759-ef9b94ef7e97"
    fake = SimpleNamespace(
        list=mock.Mock(
            return_value={
                "items": [
                    {
                        "id": project_id,
                        "title": "LeNet-5",
                        "status": "trained",
                        "latest_version_number": "v1.0.8",
                        "updated_at": "2026-05-11T03:01:26",
                    }
                ],
                "total": 1,
            }
        ),
        get=mock.Mock(return_value={"id": "p1"}),
        create=mock.Mock(return_value={"id": "p1"}),
        delete=mock.Mock(return_value=None),
        duplicate=mock.Mock(return_value={"id": "p2"}),
    )
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "list"])
        run_cli(["projects", "get", "p1"])
        run_cli(["projects", "create", "--title", "x"])
        run_cli(["projects", "delete", "p1"])
        run_cli(["projects", "duplicate", "p1", "--title", "copy"])
    out = capsys.readouterr().out
    assert project_id in out
    assert "LeNet-5" in out
    assert "trained" in out
    assert '"items"' not in out
    assert "deleted" in out


def test_projects_get_is_concise_by_default_and_verbose_json(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(
        get=mock.Mock(
            return_value={
                "id": "p1",
                "title": "LeNet-5",
                "status": "trained",
                "framework": "pytorch",
                "latest_version_number": "v1",
                "updated_at": "2026-05-26T12:00:00",
                "versions": [{"id": f"v{i}"} for i in range(20)],
            }
        ),
    )

    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "get", "p1"])
        run_cli(["projects", "get", "p1", "--verbose"])

    out = capsys.readouterr().out
    first, second = out.split("{", maxsplit=1)
    assert "Project p1" in first
    assert "versions" not in first
    assert '"versions"' in "{" + second


def test_projects_list_verbose_prints_full_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        list=mock.Mock(return_value={"items": [{"id": "p1", "title": "LeNet-5"}], "total": 1})
    )
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "list", "--verbose"])
    out = capsys.readouterr().out
    assert '"items"' in out
    assert '"title": "LeNet-5"' in out


def test_projects_list_output_saves_full_json(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    output = tmp_path / "projects.json"
    payload = {"items": [{"id": "p1", "title": "LeNet-5"}], "total": 1}
    fake = SimpleNamespace(list=mock.Mock(return_value=payload))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "list", "--output", str(output)])
    out = capsys.readouterr().out
    assert "LeNet-5" in out
    assert '"items"' not in out
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("cmd_args", "attr"),
    [
        (["projects", "list"], "list"),
        (["projects", "get", "p1"], "get"),
        (["projects", "create", "--title", "x"], "create"),
        (["projects", "delete", "p1"], "delete"),
        (["projects", "duplicate", "p1"], "duplicate"),
        (["projects", "versions", "list", "p1"], "list_versions"),
        (["projects", "versions", "get", "p1", "v1"], "get_version"),
        (["projects", "versions", "compare", "p1", "va", "vb"], "compare_versions"),
        (["projects", "versions", "restore", "p1", "v1"], "restore_version"),
        (["projects", "versions", "delete", "p1", "v1"], "delete_version"),
        (["projects", "versions", "latest", "p1"], "latest_version"),
    ],
)
def test_projects_apierrors_exit(
    run_cli: CliRunner, capsys: StrCapture, cmd_args: list[str], attr: str
) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(**{attr: mock.Mock(side_effect=APIError(500, "boom"))})
    with mock.patch("dagnam.projects", fake):
        assert run_cli(cmd_args) == 1
    err = capsys.readouterr().err
    assert "the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


# ---------------------------------------------------------------- projects architecture


def test_projects_architecture_reads_json_inputs(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    diagram = tmp_path / "diagram.json"
    config = tmp_path / "config.json"
    diagram.write_text('{"nodes": []}', encoding="utf-8")
    config.write_text('{"layers": []}', encoding="utf-8")
    save = mock.Mock(return_value={"version_id": "v1"})
    fake = SimpleNamespace(save_architecture=save)
    with mock.patch("dagnam.projects", fake):
        run_cli(
            [
                "projects",
                "architecture",
                "p1",
                "--diagram",
                f"@{diagram}",
                "--config",
                f"@{config}",
                "--message",
                "init",
            ]
        )
    save.assert_called_once_with("p1", {"nodes": []}, {"layers": []}, commit_message="init")
    out = capsys.readouterr().out
    # Concise human summary by default (not raw JSON) that surfaces the saved
    # version identifier rather than an all-dashes project summary.
    assert "Saved architecture version v1." in out
    assert '"version_id"' not in out


def test_projects_architecture_json_flag_emits_full_json(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    """``--json`` emits the full architecture response to stdout."""
    save = mock.Mock(return_value={"version_id": "v1"})
    fake = SimpleNamespace(save_architecture=save)
    with mock.patch("dagnam.projects", fake):
        run_cli(
            [
                "projects",
                "architecture",
                "p1",
                "--diagram",
                '{"nodes": []}',
                "--config",
                '{"layers": []}',
                "--json",
            ]
        )
    out = capsys.readouterr().out
    assert out.strip().startswith("{")
    assert '"version_id": "v1"' in out


def test_projects_architecture_output_saves_full_json(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    output = tmp_path / "arch.json"
    payload = {"version_id": "v1"}
    save = mock.Mock(return_value=payload)
    fake = SimpleNamespace(save_architecture=save)
    with mock.patch("dagnam.projects", fake):
        run_cli(
            [
                "projects",
                "architecture",
                "p1",
                "--diagram",
                '{"nodes": []}',
                "--config",
                '{"layers": []}',
                "--output",
                str(output),
            ]
        )
    out = capsys.readouterr().out
    assert '"version_id"' not in out
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_projects_architecture_accepts_json_literals(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    save = mock.Mock(return_value={"version_id": "v1"})
    fake = SimpleNamespace(save_architecture=save)
    with mock.patch("dagnam.projects", fake):
        run_cli(
            [
                "projects",
                "architecture",
                "p1",
                "--diagram",
                '{"nodes": [1]}',
                "--config",
                '{"layers": [2]}',
            ]
        )
    save.assert_called_once_with("p1", {"nodes": [1]}, {"layers": [2]}, commit_message=None)
    capsys.readouterr()


def test_render_architecture_falls_back_to_version_number() -> None:
    from dagnam.cli.project import _render_architecture

    assert _render_architecture({"version_number": 7}) == "Saved architecture version 7."


def test_render_architecture_handles_missing_identifiers() -> None:
    from dagnam.cli.project import _render_architecture

    assert _render_architecture({}) == "Architecture saved."
    assert _render_architecture("not-a-dict") == "Architecture saved."


def test_projects_list_empty_message(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list=mock.Mock(return_value={"items": [], "total": 0}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "list"])
    assert "No projects found." in capsys.readouterr().out


def test_projects_get_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "p.json"
    fake = SimpleNamespace(get=mock.Mock(return_value={"id": "p1", "title": "X"}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "get", "p1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "p1"


def test_projects_create_concise_by_default_and_json(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(create=mock.Mock(return_value={"id": "p1", "title": "X"}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "create", "--title", "X"])
        run_cli(["projects", "create", "--title", "X", "--json"])
    captured = capsys.readouterr()
    out = captured.out
    first, second = out.split("{", maxsplit=1)
    assert "Project p1" in first
    assert '"id": "p1"' in "{" + second
    assert "Next: dagnam training create p1 ..." in captured.err


def test_projects_duplicate_concise_by_default_and_verbose(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(duplicate=mock.Mock(return_value={"id": "p2", "title": "copy"}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "duplicate", "p1"])
        run_cli(["projects", "duplicate", "p1", "--verbose"])
    out = capsys.readouterr().out
    first, second = out.split("{", maxsplit=1)
    assert "Project p2" in first
    assert '"id": "p2"' in "{" + second


def test_projects_create_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "p.json"
    fake = SimpleNamespace(create=mock.Mock(return_value={"id": "p1"}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "create", "--title", "x", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "p1"


def test_projects_duplicate_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "p.json"
    fake = SimpleNamespace(duplicate=mock.Mock(return_value={"id": "p2"}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "duplicate", "p1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "p2"


def test_projects_architecture_bad_json_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(save_architecture=mock.Mock())
    with mock.patch("dagnam.projects", fake):
        with pytest.raises(SystemExit) as exc:
            run_cli(["projects", "architecture", "p1", "--diagram", "{bad", "--config", "{}"])
    assert exc.value.code == 1
    assert "Could not read JSON input" in capsys.readouterr().err


def test_projects_architecture_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(save_architecture=mock.Mock(side_effect=APIError(500, "boom")))
    with mock.patch("dagnam.projects", fake):
        assert run_cli(["projects", "architecture", "p1", "--diagram", "{}", "--config", "{}"]) == 1
    err = capsys.readouterr().err
    assert "the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


def test_projects_list_dict_with_non_list_items(run_cli: CliRunner, capsys: StrCapture) -> None:
    """A dict whose ``items`` is not a list renders as empty, not a crash."""
    fake = SimpleNamespace(list=mock.Mock(return_value={"items": None, "total": 0}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "list"])
    assert "No projects found." in capsys.readouterr().out


def test_projects_list_bare_list_result(run_cli: CliRunner, capsys: StrCapture) -> None:
    """A bare list result (not a paginated dict) renders via the list branch."""
    fake = SimpleNamespace(list=mock.Mock(return_value=[{"id": "p1", "title": "Solo"}]))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "list"])
    assert "Solo" in capsys.readouterr().out


# ---------------------------------------------------------------- projects versions


def test_projects_versions_list(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        list_versions=mock.Mock(
            return_value={"items": [{"id": "v1", "version_number": "1.0.0"}], "total": 1}
        )
    )
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "versions", "list", "p1", "--page", "2", "--limit", "5"])
    assert "v1" in capsys.readouterr().out
    fake.list_versions.assert_called_once_with("p1", page=2, limit=5)


def test_projects_versions_get(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        get_version=mock.Mock(return_value={"id": "v1", "version_number": "1.0.0"})
    )
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "versions", "get", "p1", "v1"])
    assert "1.0.0" in capsys.readouterr().out
    fake.get_version.assert_called_once_with("p1", "v1")


def test_projects_versions_compare(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        compare_versions=mock.Mock(return_value={"version_a": {}, "version_b": {}})
    )
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "versions", "compare", "p1", "va", "vb"])
    assert "version_a" in capsys.readouterr().out
    fake.compare_versions.assert_called_once_with("p1", "va", "vb")


def test_projects_versions_restore(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(restore_version=mock.Mock(return_value={"id": "v2", "is_current": True}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "versions", "restore", "p1", "v1"])
    assert "v2" in capsys.readouterr().out
    fake.restore_version.assert_called_once_with("p1", "v1")


def test_projects_versions_delete(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete_version=mock.Mock(return_value=None))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "versions", "delete", "p1", "v1"])
    assert "v1" in capsys.readouterr().out
    fake.delete_version.assert_called_once_with("p1", "v1")


def test_projects_versions_latest(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(latest_version=mock.Mock(return_value={"id": "v2", "is_current": True}))
    with mock.patch("dagnam.projects", fake):
        run_cli(["projects", "versions", "latest", "p1"])
    assert "v2" in capsys.readouterr().out
    fake.latest_version.assert_called_once_with("p1")


# ------------------------------------------------------------- W6 write commands


def test_projects_update(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.projects.update", return_value={"id": "p1", "title": "New"}) as m:
        assert (
            run_cli(["projects", "update", "p1", "--title", "New", "--visibility", "public"]) == 0
        )
    m.assert_called_once_with("p1", title="New", visibility="public")
    assert json.loads(capsys.readouterr().out)["title"] == "New"


def test_projects_update_requires_at_least_one_field(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with mock.patch("dagnam.projects.update") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(["projects", "update", "p1"])
    assert exc_info.value.code == 1
    m.assert_not_called()
    assert "Nothing to update" in capsys.readouterr().err


def test_projects_update_parses_tags_csv(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.projects.update", return_value={}) as m:
        run_cli(["projects", "update", "p1", "--tags", "a,b"])
    m.assert_called_once_with("p1", tags=["a", "b"])


def test_projects_update_description_and_framework(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.projects.update", return_value={}) as m:
        run_cli(["projects", "update", "p1", "--description", "d2", "--framework", "jax"])
    m.assert_called_once_with("p1", description="d2", framework="jax")


def test_projects_bulk_delete_with_yes(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.projects.bulk_delete", return_value={"deleted": 2}) as m:
        assert run_cli(["projects", "bulk-delete", "p1", "p2", "--yes"]) == 0
    m.assert_called_once_with(["p1", "p2"])
    assert "2" in capsys.readouterr().out


def test_projects_bulk_delete_typed_confirmation(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda *a: "delete 2")
    with mock.patch("dagnam.projects.bulk_delete", return_value={"deleted": 2}) as m:
        assert run_cli(["projects", "bulk-delete", "p1", "p2"]) == 0
    m.assert_called_once()


def test_projects_bulk_delete_wrong_confirmation_aborts(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda *a: "delete everything")
    with mock.patch("dagnam.projects.bulk_delete") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(["projects", "bulk-delete", "p1", "p2"])
    assert exc_info.value.code == 1
    m.assert_not_called()


def test_projects_link_dataset(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.projects.link_dataset", return_value={"linked": True}) as m:
        assert run_cli(["projects", "link-dataset", "p1", "ds1", "--role", "train"]) == 0
    m.assert_called_once_with("p1", "ds1", "train")


def test_projects_unlink_dataset(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.projects.unlink_dataset", return_value=None) as m:
        assert run_cli(["projects", "unlink-dataset", "p1", "ds1"]) == 0
    m.assert_called_once_with("p1", "ds1")
    assert "unlinked" in capsys.readouterr().out.lower()
