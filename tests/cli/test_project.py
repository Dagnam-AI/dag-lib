"""CLI projects subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


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
    ],
)
def test_projects_apierrors_exit(run_cli: CliRunner, cmd_args: list[str], attr: str) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(**{attr: mock.Mock(side_effect=APIError(500, "boom"))})
    with mock.patch("dagnam.projects", fake):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)


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
    assert '"version_id": "v1"' in capsys.readouterr().out


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
