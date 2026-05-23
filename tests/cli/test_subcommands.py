"""End-to-end CLI coverage via argparse + main() with mocked facades."""

from __future__ import annotations
from pathlib import Path
from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


import json
import argparse
from types import SimpleNamespace
from unittest import mock

import pytest

from dagnam.cli import main as cli_main
from dagnam.cli import login as login_mod


def _bad_key_prompt(_prompt: str) -> str:
    return "bad-key"


def _good_key_prompt(_prompt: str) -> str:
    return "good-key"


def _key_prompt(_prompt: str) -> str:
    return "k"


@pytest.fixture
def run_cli(monkeypatch: PytestMonkeyPatch):
    """Helper to set sys.argv and invoke main(); returns nothing — use capsys."""

    def _run(argv: list[str]) -> None:
        monkeypatch.setattr("sys.argv", ["dagnam", *argv])
        cli_main()

    return _run


# ---------------------------------------------------------------- subcommand-help


@pytest.mark.parametrize(
    "subcmd",
    ["dataset", "cache", "inference", "checkpoint", "deployments", "hub", "projects", "codegen"],
)
def test_subcommand_without_action_prints_help_and_exits(subcmd: str, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["dagnam", subcmd])
    with pytest.raises(SystemExit):
        cli_main()


# ---------------------------------------------------------------- dataset


def test_dataset_list_empty(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["dataset", "list"])
    assert "No datasets found" in capsys.readouterr().out


def test_dataset_list_with_rows(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.list_datasets",
        return_value=[
            {
                "id": "ds-1",
                "name": "Iris",
                "format": "csv",
                "num_samples": 150,
                "dataset_type": "tabular",
            }
        ],
    ):
        run_cli(["dataset", "list"])
    out = capsys.readouterr().out
    assert "ds-1" in out
    assert "Iris" in out


def test_dataset_list_autherror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch, tmp_path: Path) -> None:
    from pathlib import Path

    monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
    # Redirect config to an empty location so auth resolution fails.
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", Path(tmp_path) / "missing.json")
    with pytest.raises(SystemExit):
        run_cli(["dataset", "list"])


def test_dataset_list_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_datasets",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "list"])


def test_dataset_info(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        return_value={
            "id": "ds-1",
            "schema": {"col1": "int"},
            "class_names": ["a", "b"],
        },
    ):
        run_cli(["dataset", "info", "ds-1"])
    out = capsys.readouterr().out
    assert "id: ds-1" in out
    assert "col1: int" in out
    assert "class_names: a, b" in out


def test_dataset_info_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "info", "ds-1"])


def test_dataset_download(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam.load_dataset", return_value=None):
        run_cli(["dataset", "download", "ds-1"])
    assert "downloaded" in capsys.readouterr().out


def test_dataset_download_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.load_dataset", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "download", "ds-1"])


# ---------------------------------------------------------------- cache


def test_cache_clear(run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    (cache_dir / "ds-1").mkdir()
    (cache_dir / "ds-1" / "data").write_text("x")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)
    run_cli(["cache", "clear"])
    out = capsys.readouterr().out
    assert "Cleared" in out or "cleared" in out.lower()


def test_cache_list_with_entries(run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    sub = cache_dir / "ds-1"
    sub.mkdir()
    (sub / "data").write_bytes(b"hi")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)
    run_cli(["cache", "list"])
    out = capsys.readouterr().out
    assert "ds-1" in out


# ---------------------------------------------------------------- inference


def test_inference_run(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.inference", return_value={"label": "ok"}):
        run_cli(["inference", "run", "dep-1", "--input", '{"x":1}'])
    assert json.loads(capsys.readouterr().out) == {"label": "ok"}


def test_inference_run_bad_json(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run_cli(["inference", "run", "dep-1", "--input", "not-json"])


def test_inference_run_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.inference", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["inference", "run", "dep-1", "--input", '{"x":1}'])


def test_inference_batch(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.inference_batch", return_value=[{"y": 1}, {"y": 2}]):
        run_cli(["inference", "batch", "dep-1", "--inputs", '[{"x":1},{"x":2}]'])
    assert json.loads(capsys.readouterr().out) == [{"y": 1}, {"y": 2}]


def test_inference_batch_rejects_non_array(run_cli: CliRunner) -> None:
    with pytest.raises(SystemExit):
        run_cli(["inference", "batch", "dep-1", "--inputs", '{"x":1}'])


def test_inference_batch_bad_json(run_cli: CliRunner) -> None:
    with pytest.raises(SystemExit):
        run_cli(["inference", "batch", "dep-1", "--inputs", "garbage"])


def test_inference_batch_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.inference_batch", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["inference", "batch", "dep-1", "--inputs", "[1,2]"])


def test_inference_batch_loads_from_file(run_cli: CliRunner, tmp_path: Path, capsys: StrCapture) -> None:
    fp = tmp_path / "data.json"
    fp.write_text("[1, 2, 3]")
    with mock.patch("dagnam.inference_batch", return_value=[1, 2, 3]):
        run_cli(["inference", "batch", "dep-1", "--inputs", f"@{fp}"])
    assert "1" in capsys.readouterr().out


def test_inference_batch_file_readerror_exits(run_cli: CliRunner, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run_cli(
            [
                "inference",
                "batch",
                "dep-1",
                "--inputs",
                f"@{tmp_path / 'does-not-exist.json'}",
            ]
        )


def test_inference_health(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.deployment_health", return_value={"status": "healthy"}):
        run_cli(["inference", "health", "dep-1"])
    assert json.loads(capsys.readouterr().out) == {"status": "healthy"}


def test_inference_health_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.deployment_health", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["inference", "health", "dep-1"])


# ---------------------------------------------------------------- checkpoint


def test_checkpoint_list_empty(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.list_checkpoints", return_value=[]):
        run_cli(["checkpoint", "list", "job-1"])
    assert "No checkpoints" in capsys.readouterr().out


def test_checkpoint_list_with_rows(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        return_value=[
            {
                "id": "ck-1",
                "epoch": 3,
                "step": 100,
                "is_best": True,
                "is_final": False,
                "file_size": 2048,
            }
        ],
    ):
        run_cli(["checkpoint", "list", "job-1"])
    out = capsys.readouterr().out
    assert "ck-1" in out
    assert "True" in out


def test_checkpoint_list_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_checkpoints",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["checkpoint", "list", "job-1"])


def test_checkpoint_download(run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path"):
        run_cli(["checkpoint", "download", "job-1"])
    assert "/some/path" in capsys.readouterr().out


def test_checkpoint_download_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.download_checkpoint", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["checkpoint", "download", "job-1"])


# ---------------------------------------------------------------- stream


def test_stream_emits_human_readable(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake_event = SimpleNamespace(event="progress", data={"step": 1}, id="e1", retry=None)
    with mock.patch("dagnam.stream_training", return_value=iter([fake_event])):
        run_cli(["stream", "job-1"])
    assert "[progress]" in capsys.readouterr().out


def test_stream_json_mode(run_cli: CliRunner, capsys: StrCapture) -> None:
    # asdict requires a dataclass; use the real SSEEvent.
    from dagnam._core.sse import SSEEvent

    ev = SSEEvent(event="progress", data={"step": 1}, id="e1", retry=None)
    with mock.patch("dagnam.stream_training", return_value=iter([ev])):
        run_cli(["stream", "job-1", "--json"])
    assert json.loads(capsys.readouterr().out.strip()) == {
        "event": "progress",
        "data": {"step": 1},
        "id": "e1",
        "retry": None,
    }


def test_stream_keyboard_interrupt_exits_130(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.stream_training", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["stream", "job-1"])
    assert exc_info.value.code == 130


def test_stream_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.stream_training", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["stream", "job-1"])


# ---------------------------------------------------------------- deployments


def test_deployments_list(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list=mock.Mock(return_value={"items": []}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "list"])
    assert "items" in capsys.readouterr().out


def test_deployments_get(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(get=mock.Mock(return_value={"id": "dep-1"}))
    with mock.patch("dagnam.deployments", fake):
        run_cli(["deployments", "get", "dep-1"])
    assert "dep-1" in capsys.readouterr().out


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


# ---------------------------------------------------------------- hub


def test_hub_search(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(search=mock.Mock(return_value={"items": []}))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "search", "--search", "vit"])
    assert "items" in capsys.readouterr().out


def test_hub_get(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(get=mock.Mock(return_value={"id": "m1"}))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "get", "m1"])
    assert "m1" in capsys.readouterr().out


def test_hub_star_unstar_fork_featured_trending(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        star=mock.Mock(return_value={"ok": True}),
        unstar=mock.Mock(return_value={"ok": True}),
        fork=mock.Mock(return_value={"id": "m2"}),
        featured=mock.Mock(return_value=["a"]),
        trending=mock.Mock(return_value=["t"]),
    )
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "star", "m1"])
        run_cli(["hub", "unstar", "m1"])
        run_cli(["hub", "fork", "m1"])
        run_cli(["hub", "featured"])
        run_cli(["hub", "trending", "--days", "14"])
    capsys.readouterr()


@pytest.mark.parametrize(
    ("cmd_args", "attr"),
    [
        (["hub", "search"], "search"),
        (["hub", "get", "m1"], "get"),
        (["hub", "star", "m1"], "star"),
        (["hub", "unstar", "m1"], "unstar"),
        (["hub", "fork", "m1"], "fork"),
        (["hub", "featured"], "featured"),
        (["hub", "trending"], "trending"),
    ],
)
def test_hub_apierrors_exit(run_cli: CliRunner, cmd_args: list[str], attr: str) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(**{attr: mock.Mock(side_effect=APIError(500, "boom"))})
    with mock.patch("dagnam.hub", fake):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)


# ---------------------------------------------------------------- projects


def test_projects_list_get_create_delete_duplicate(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        list=mock.Mock(return_value={"items": []}),
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
    assert "deleted" in out


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


# ---------------------------------------------------------------- codegen


def test_codegen_generate_preview_validate_download(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        generate=mock.Mock(return_value={"task_id": "t1"}),
        preview=mock.Mock(return_value={"code": "..."}),
        validate=mock.Mock(return_value={"valid": True}),
        download=mock.Mock(return_value="/tmp/code.zip"),
    )
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "generate", "p1", "--framework", "pytorch"])
        run_cli(["codegen", "preview", "p1", "--framework", "pytorch"])
        run_cli(["codegen", "validate", "p1"])
        run_cli(["codegen", "download", "p1", "--framework", "pytorch", "--output", "/tmp/out"])
    capsys.readouterr()


@pytest.mark.parametrize(
    ("cmd_args", "attr"),
    [
        (["codegen", "generate", "p1"], "generate"),
        (["codegen", "preview", "p1"], "preview"),
        (["codegen", "validate", "p1"], "validate"),
        (["codegen", "download", "p1"], "download"),
    ],
)
def test_codegen_apierrors_exit(run_cli: CliRunner, cmd_args: list[str], attr: str) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(**{attr: mock.Mock(side_effect=APIError(500, "boom"))})
    with mock.patch("dagnam.codegen", fake):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)


# ---------------------------------------------------------------- login


def test_web_url_from_api_url_prod() -> None:
    assert login_mod._web_url_from_api_url("https://api.dagnam.ai") == "https://dagnam.ai"


def test_web_url_from_api_url_local() -> None:
    assert login_mod._web_url_from_api_url("http://localhost:8000") == "http://localhost:5173"


def test_web_url_from_api_url_unknown() -> None:
    assert login_mod._web_url_from_api_url("https://corp.internal") == ""


def test_login_prints_help_block(capsys: StrCapture, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(login_mod, "error", lambda msg: (_ for _ in ()).throw(SystemExit(msg)))

    ns = argparse.Namespace(api_url="http://localhost:8000")
    with pytest.raises(SystemExit):
        login_mod.cmd_login(ns, getpass_func=lambda _p: "sk_will_fail")
    out = capsys.readouterr().out
    assert "Don't have an API key yet?" in out
    assert "http://localhost:5173" in out


def test_login_apierror_exits(run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _bad_key_prompt)
    from dagnam._core.exceptions import AuthError

    with mock.patch(
        "dagnam._core.client.DagnamClient.list_datasets",
        side_effect=AuthError("invalid"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["login"])


def test_login_preserves_existing_config(run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"other": "kept"}))
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _good_key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login"])
    data = json.loads(config_file.read_text())
    assert data["api_key"] == "good-key"
    assert data["other"] == "kept"


def test_login_with_custom_api_url(run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login", "--api-url", "https://custom"])
    data = json.loads(config_file.read_text())
    assert data["api_url"] == "https://custom"


def test_login_corrupt_existing_config_starts_fresh(run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    config_dir = tmp_path / ".dagnam"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text("not json {{{")
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login"])
    data = json.loads(config_file.read_text())
    assert data == {"api_key": "k"}
