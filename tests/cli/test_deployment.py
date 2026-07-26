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
    captured = capsys.readouterr()
    assert '"id": "dep-1"' in captured.out
    assert "Next: dagnam inference dep-1 run ..." in captured.err


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
        ["deployments", "retry", "x"],
        ["deployments", "delete", "x"],
        ["deployments", "logs", "x"],
        ["deployments", "metrics", "x"],
        ["deployments", "platforms"],
        ["deployments", "estimate-cost", "--platform", "aws", "--instance-type", "s"],
        [
            "deployments",
            "validate",
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
    ],
)
def test_deployments_apierrors_exit(
    run_cli: CliRunner, capsys: StrCapture, cmd_args: list[str]
) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(
        list=mock.Mock(side_effect=APIError(500, "boom")),
        get=mock.Mock(side_effect=APIError(500, "boom")),
        create=mock.Mock(side_effect=APIError(500, "boom")),
        pause=mock.Mock(side_effect=APIError(500, "boom")),
        resume=mock.Mock(side_effect=APIError(500, "boom")),
        retry=mock.Mock(side_effect=APIError(500, "boom")),
        delete=mock.Mock(side_effect=APIError(500, "boom")),
        logs=mock.Mock(side_effect=APIError(500, "boom")),
        metrics=mock.Mock(side_effect=APIError(500, "boom")),
        platforms=mock.Mock(side_effect=APIError(500, "boom")),
        estimate_cost=mock.Mock(side_effect=APIError(500, "boom")),
        validate=mock.Mock(side_effect=APIError(500, "boom")),
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


# ---------------------------------------------------------------- planning


def test_deployments_platforms(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        platforms=mock.Mock(return_value=[{"platform": "fastapi", "name": "FastAPI"}])
    )
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "platforms"])
    assert "fastapi" in capsys.readouterr().out
    fake.platforms.assert_called_once_with()


def test_deployments_retry(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(retry=mock.Mock(return_value={"id": "d1", "status": "deploying"}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "retry", "d1"])
    out = capsys.readouterr().out
    assert "d1" in out
    assert "deploying" in out
    fake.retry.assert_called_once_with("d1")


def test_deployments_estimate_cost(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(estimate_cost=mock.Mock(return_value={"monthly_cost": 12.0}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(
            [
                "deployments",
                "estimate-cost",
                "--platform",
                "fastapi",
                "--instance-type",
                "cpu.small",
                "--auto-scaling",
                "--min-instances",
                "1",
                "--max-instances",
                "3",
                "--region",
                "us-east-1",
            ]
        )
    assert "monthly_cost" in capsys.readouterr().out
    kwargs = fake.estimate_cost.call_args.kwargs
    assert kwargs["platform"] == "fastapi"
    assert kwargs["instance_type"] == "cpu.small"
    assert kwargs["auto_scaling_enabled"] is True
    assert kwargs["min_instances"] == 1
    assert kwargs["max_instances"] == 3
    assert kwargs["region"] == "us-east-1"


def test_deployments_validate(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(validate=mock.Mock(return_value={"valid": True, "errors": []}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(
            [
                "deployments",
                "validate",
                "--name",
                "x",
                "--project-id",
                "p1",
                "--checkpoint-path",
                "/c.pt",
                "--platform",
                "fastapi",
                "--deployment-type",
                "text",
                "--instance-type",
                "cpu.small",
            ]
        )
    assert "valid" in capsys.readouterr().out
    kwargs = fake.validate.call_args.kwargs
    assert kwargs["name"] == "x"
    assert kwargs["deployment_type"] == "text"


def test_deployments_collect_metrics(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch(
        "dagnam.deployments.collect_metrics",
        return_value={"deployment_id": "dep-1", "points_created": 60, "backfilled": True},
    ) as m:
        assert run_cli(["deployments", "collect-metrics", "dep-1", "--backfill-minutes", "90"]) == 0
    m.assert_called_once_with("dep-1", backfill_minutes=90)
    assert json.loads(capsys.readouterr().out)["points_created"] == 60


# ---------------------------------------------------------------- update / scale / rollback


def test_deployments_update(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.deployments.update", return_value={"id": "dep-1"}) as m:
        assert run_cli(["deployments", "update", "dep-1", "--name", "n2", "--auto-scaling"]) == 0
    m.assert_called_once_with("dep-1", name="n2", auto_scaling_enabled=True)


def test_deployments_update_no_auto_scaling_flag(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.deployments.update", return_value={}) as m:
        run_cli(["deployments", "update", "dep-1", "--no-auto-scaling"])
    m.assert_called_once_with("dep-1", auto_scaling_enabled=False)


def test_deployments_update_all_fields(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.deployments.update", return_value={}) as m:
        run_cli(
            [
                "deployments",
                "update",
                "dep-1",
                "--instance-type",
                "t3.large",
                "--num-instances",
                "4",
                "--min-instances",
                "2",
                "--max-instances",
                "8",
            ]
        )
    m.assert_called_once_with(
        "dep-1",
        instance_type="t3.large",
        num_instances=4,
        min_instances=2,
        max_instances=8,
    )


def test_deployments_update_requires_a_field(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.deployments.update") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(["deployments", "update", "dep-1"])
    assert exc_info.value.code == 1
    m.assert_not_called()
    assert "Nothing to update" in capsys.readouterr().err


def test_deployments_scale_waits(run_cli: CliRunner, capsys: StrCapture) -> None:
    op = mock.MagicMock()
    op.wait.return_value.result.return_value = {"id": "dep-1", "num_instances": 3}
    with mock.patch("dagnam.deployments.scale", return_value=op) as m:
        assert run_cli(["deployments", "scale", "dep-1", "--num-instances", "3"]) == 0
    m.assert_called_once_with("dep-1", 3)
    op.wait.assert_called_once()
    assert json.loads(capsys.readouterr().out)["num_instances"] == 3


def test_deployments_scale_no_wait(run_cli: CliRunner, capsys: StrCapture) -> None:
    op = mock.MagicMock()
    op.initial.return_value = {"id": "dep-1", "status": "scaling"}
    with mock.patch("dagnam.deployments.scale", return_value=op):
        assert run_cli(["deployments", "scale", "dep-1", "--num-instances", "2", "--no-wait"]) == 0
    op.wait.assert_not_called()
    assert json.loads(capsys.readouterr().out)["status"] == "scaling"


def test_deployments_rollback_waits(run_cli: CliRunner, capsys: StrCapture) -> None:
    op = mock.MagicMock()
    op.wait.return_value.result.return_value = {"id": "dep-1", "status": "running"}
    with mock.patch("dagnam.deployments.rollback", return_value=op) as m:
        assert run_cli(["deployments", "rollback", "dep-1", "--checkpoint-id", "ckpt-best"]) == 0
    m.assert_called_once_with("dep-1", "ckpt-best")
    assert json.loads(capsys.readouterr().out)["status"] == "running"


def test_deployments_rollback_no_wait(run_cli: CliRunner, capsys: StrCapture) -> None:
    op = mock.MagicMock()
    op.initial.return_value = {"id": "dep-1", "status": "rolling_back"}
    with mock.patch("dagnam.deployments.rollback", return_value=op):
        assert (
            run_cli(["deployments", "rollback", "dep-1", "--checkpoint-id", "ckpt-x", "--no-wait"])
            == 0
        )
    op.wait.assert_not_called()
    assert json.loads(capsys.readouterr().out)["status"] == "rolling_back"
