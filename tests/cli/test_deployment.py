"""CLI deployments subcommand."""

from __future__ import annotations

import json
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


def test_deployments_deploy_version_prints_the_key_once(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    fake = SimpleNamespace(
        deploy_model_version=mock.Mock(
            return_value={"id": "dep-1", "status": "not_provisioned", "api_key": "dep-secret"}
        )
    )
    with mock.patch("dagnam.deployments", fake):
        assert run_cli(["deployments", "deploy-version", "mv-1"]) == 0
    fake.deploy_model_version.assert_called_once_with("mv-1", name=None, project_id=None)
    captured = capsys.readouterr()
    assert captured.out.count("dep-secret") == 1
    assert "will not be shown again" in captured.out
    assert "Next: dagnam deployments revisions dep-1" in captured.err


def test_deployments_deploy_version_forwards_flags_and_saves_json(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    out_path = tmp_path / "deployment.json"
    fake = SimpleNamespace(
        deploy_model_version=mock.Mock(return_value={"id": "dep-1", "api_key": "dep-secret"})
    )
    with mock.patch("dagnam.deployments", fake):
        run_cli(
            [
                "deployments",
                "deploy-version",
                "mv-1",
                "--name",
                "bot",
                "--project-id",
                "p1",
                "--json",
                "--output",
                str(out_path),
            ]
        )
    fake.deploy_model_version.assert_called_once_with("mv-1", name="bot", project_id="p1")
    assert json.loads(capsys.readouterr().out)["api_key"] == "dep-secret"
    assert json.loads(out_path.read_text(encoding="utf-8"))["api_key"] == "dep-secret"


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
        ["deployments", "deploy-version", "mv-1"],
        ["deployments", "pause", "x"],
        ["deployments", "resume", "x"],
        ["deployments", "delete", "x"],
        ["deployments", "logs", "x"],
        ["deployments", "metrics", "x"],
        ["deployments", "revisions", "x"],
        ["deployments", "update", "x", "--name", "n"],
    ],
)
def test_deployments_apierrors_exit(
    run_cli: CliRunner, capsys: StrCapture, cmd_args: list[str]
) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        list=mock.Mock(side_effect=APIError(500, "boom")),
        get=mock.Mock(side_effect=APIError(500, "boom")),
        deploy_model_version=mock.Mock(side_effect=APIError(500, "boom")),
        pause=mock.Mock(side_effect=APIError(500, "boom")),
        resume=mock.Mock(side_effect=APIError(500, "boom")),
        delete=mock.Mock(side_effect=APIError(500, "boom")),
        logs=mock.Mock(side_effect=APIError(500, "boom")),
        metrics=mock.Mock(side_effect=APIError(500, "boom")),
        revisions=mock.Mock(side_effect=APIError(500, "boom")),
        update=mock.Mock(side_effect=APIError(500, "boom")),
    )
    with mock.patch("dagnam.deployments", fake):
        assert run_cli(cmd_args) == 1
    err = capsys.readouterr().err
    assert "the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


def test_deployments_list_empty_message(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list=mock.Mock(return_value={"items": [], "total": 0}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list"])
    assert "No deployments found." in capsys.readouterr().out


def test_deployments_list_non_dict_result_passthrough(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    # A bare list (not a paginated dict) exercises the list-branch of the helpers.
    fake = SimpleNamespace(list=mock.Mock(return_value=[{"id": "dep-1", "name": "api"}]))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list"])
    assert "dep-1" in capsys.readouterr().out


def test_deployments_get_non_dict_passthrough(run_cli: CliRunner, capsys: StrCapture) -> None:
    """A non-dict deployment payload is returned unchanged by the redactor."""
    fake = SimpleNamespace(get=mock.Mock(return_value=["not", "a", "dict"]))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "get", "dep-1"])
    assert "not" in capsys.readouterr().out


def test_deployments_list_dict_without_list_items(run_cli: CliRunner, capsys: StrCapture) -> None:
    """A dict result whose ``items`` is not a list skips per-item redaction."""
    fake = SimpleNamespace(list=mock.Mock(return_value={"items": None, "total": 0}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list"])
    assert "No deployments found." in capsys.readouterr().out


# ---------------------------------------------------------------- update


def test_deployments_update_renames(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.deployments.update", return_value={"id": "dep-1"}) as m:
        assert run_cli(["deployments", "update", "dep-1", "--name", "n2"]) == 0
    m.assert_called_once_with("dep-1", name="n2")
    assert "dep-1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra", [[], ["--num-instances", "4"], ["--instance-type", "t3.large"], ["--auto-scaling"]]
)
def test_deployments_update_takes_only_a_name(
    run_cli: CliRunner, capsys: StrCapture, extra: list[str]
) -> None:
    # Serving capacity is managed by the platform, so a name is all there is to change.
    with mock.patch("dagnam.deployments.update") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(["deployments", "update", "dep-1", *extra])
    assert exc_info.value.code == 2
    m.assert_not_called()


def test_deployments_revisions(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(revisions=mock.Mock(return_value=[{"id": "rev1"}]))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "revisions", "dep-1", "--page", "2", "--limit", "5"])
    assert "rev1" in capsys.readouterr().out
    fake.revisions.assert_called_once_with("dep-1", page=2, limit=5)


@pytest.mark.parametrize(
    "command",
    [
        "create",
        "scale",
        "rollback",
        "retry",
        "estimate-cost",
        "collect-metrics",
        "platforms",
        "validate",
    ],
)
def test_removed_deployment_commands_are_unknown(
    run_cli: CliRunner, capsys: StrCapture, command: str
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["deployments", command, "--help"])
    assert exc_info.value.code == 2
    assert f"unknown subcommand '{command}'" in capsys.readouterr().err
