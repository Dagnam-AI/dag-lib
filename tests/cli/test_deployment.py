"""CLI deployments subcommand."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


# ---------------------------------------------------------------- deployments


def test_deployments_list(run_cli: CliRunner, capsys: StrCapture) -> None:
    deployment_id = "dep-123456-7890"
    fake = SimpleNamespace(
        list=mock.Mock(
            return_value={
                "items": [
                    {
                        "id": deployment_id,
                        "name": "api-prod",
                        "status": "running",
                        "platform": "aws",
                        "updated_at": "2026-05-20T12:34:56",
                    }
                ],
                "total": 1,
            }
        )
    )
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list"])
    out = capsys.readouterr().out
    assert deployment_id in out
    assert "api-prod" in out
    assert "running" in out
    assert '"items"' not in out


def test_deployments_list_verbose_prints_full_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        list=mock.Mock(return_value={"items": [{"id": "dep-1", "name": "api-prod"}], "total": 1})
    )
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list", "--verbose"])
    out = capsys.readouterr().out
    assert '"items"' in out
    assert '"name": "api-prod"' in out


def test_deployments_list_json_redacts_serving_key(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    output = tmp_path / "deployments.json"
    fake = SimpleNamespace(
        list=mock.Mock(
            return_value={"items": [{"id": "dep-1", "api_key": "serving-secret"}], "total": 1}
        )
    )
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list", "--json", "--output", str(output)])

    stdout = capsys.readouterr().out
    saved = output.read_text(encoding="utf-8")
    assert "serving-secret" not in stdout
    assert "serving-secret" not in saved
    assert "<redacted>" in stdout
    assert "<redacted>" in saved


def test_deployments_list_prints_pagination_footer(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        list=mock.Mock(return_value={"items": [{"id": "dep-1"}], "total": 3, "page": 1, "pages": 3})
    )
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list"])
    assert "Page 1 of 3 - showing 1 of 3" in capsys.readouterr().out


def test_deployments_get(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(get=mock.Mock(return_value={"id": "dep-1", "api_key": "secret"}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "get", "dep-1"])
    out = capsys.readouterr().out
    assert "dep-1" in out
    assert "<redacted>" in out
    assert "secret" not in out


def test_deployments_create_wait_result(run_cli: CliRunner, capsys: StrCapture) -> None:
    create_chain = mock.Mock()
    create_chain.wait.return_value.result.return_value = {"id": "dep-1"}
    fake = SimpleNamespace(create=mock.Mock(return_value=create_chain))
    with mock.patch("dagnam.deployments", fake):
        run_cli(
            [
                "deployments",
                "create",
                "--name",
                "x",
                "--project-id",
                "p1",
                "--checkpoint-path",
                "ck/p",
                "--platform",
                "aws",
                "--deployment-type",
                "production",
                "--instance-type",
                "small",
            ]
        )
    assert '"id": "dep-1"' in capsys.readouterr().out


def test_deployments_pause(run_cli: CliRunner, capsys: StrCapture) -> None:
    chain = mock.Mock()
    chain.wait.return_value = None
    fake = SimpleNamespace(pause=mock.Mock(return_value=chain))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "pause", "dep-1"])
    assert "paused" in capsys.readouterr().out


def test_deployments_resume(run_cli: CliRunner, capsys: StrCapture) -> None:
    chain = mock.Mock()
    chain.wait.return_value = None
    fake = SimpleNamespace(resume=mock.Mock(return_value=chain))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "resume", "dep-1"])
    assert "resumed" in capsys.readouterr().out


def test_deployments_delete(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete=mock.Mock(return_value=None))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "delete", "dep-1"])
    assert "deleted" in capsys.readouterr().out


def test_deployments_logs(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(logs=mock.Mock(return_value={"items": []}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "logs", "dep-1", "--level", "ERROR"])
    assert "items" in capsys.readouterr().out


def test_deployments_metrics(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(metrics=mock.Mock(return_value={"qps": 1}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "metrics", "dep-1"])
    assert "qps" in capsys.readouterr().out


@pytest.mark.parametrize(
    "cmd_args",
    [
        ["deployments", "list"],
        ["deployments", "get", "x"],
        [
            "deployments",
            "create",
            "--name",
            "x",
            "--project-id",
            "p",
            "--checkpoint-path",
            "c",
            "--platform",
            "aws",
            "--deployment-type",
            "production",
            "--instance-type",
            "s",
        ],
        ["deployments", "pause", "x"],
        ["deployments", "resume", "x"],
        ["deployments", "delete", "x"],
        ["deployments", "logs", "x"],
        ["deployments", "metrics", "x"],
    ],
)
def test_deployments_apierrors_exit(run_cli: CliRunner, cmd_args: list[str]) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        list=mock.Mock(side_effect=APIError(500, "boom")),
        get=mock.Mock(side_effect=APIError(500, "boom")),
        create=mock.Mock(side_effect=APIError(500, "boom")),
        pause=mock.Mock(side_effect=APIError(500, "boom")),
        resume=mock.Mock(side_effect=APIError(500, "boom")),
        delete=mock.Mock(side_effect=APIError(500, "boom")),
        logs=mock.Mock(side_effect=APIError(500, "boom")),
        metrics=mock.Mock(side_effect=APIError(500, "boom")),
    )
    with mock.patch("dagnam.deployments", fake):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)
