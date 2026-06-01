"""End-to-end CLI coverage via argparse + main() with mocked facades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from dagnam.cli import login as login_mod, main as cli_main

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


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
def test_subcommand_without_action_prints_help_and_exits(
    subcmd: str, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["dagnam", subcmd])
    with pytest.raises(SystemExit):
        cli_main()


def test_top_level_help_includes_workflow_examples(run_cli: CliRunner, capsys: StrCapture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "dagnam training attach" in out
    assert "dagnam config set training_metrics_path" in out


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["dataset", "list", "--help"], "image, text, audio, video, tabular, custom"),
        (["deployments", "create", "--help"], "fastapi, torchserve, vllm, triton, custom"),
        (["deployments", "logs", "--help"], "debug, info, warning, error"),
        (["inference", "run", "--help"], "--input-file"),
        (["inference", "batch", "--help"], "--inputs-file"),
    ],
)
def test_help_documents_supported_values_and_file_inputs(
    run_cli: CliRunner, capsys: StrCapture, args: list[str], expected: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(args)
    assert excinfo.value.code == 0
    normalized = " ".join(capsys.readouterr().out.split()).replace(" or ", " ")
    assert expected in normalized


# ---------------------------------------------------------------- dataset


def test_dataset_list_empty(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["dataset", "list"])
    assert "No datasets found" in capsys.readouterr().out


def test_dataset_list_with_rows(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
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


def test_dataset_list_json_forwards_filters_and_overrides(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    list_datasets = mock.Mock(return_value=[{"id": "ds-1"}])
    with mock.patch("dagnam._core.client.DagnamClient") as client:
        client.return_value.list_datasets = list_datasets
        run_cli(
            [
                "dataset",
                "list",
                "--type",
                "tabular",
                "--search",
                "iris",
                "--api-url",
                "https://example.test",
                "--api-key",
                "key",
                "--json",
            ]
        )
    client.assert_called_once_with("https://example.test", "key")
    list_datasets.assert_called_once_with(type="tabular", search="iris")
    assert json.loads(capsys.readouterr().out) == [{"id": "ds-1"}]


def test_dataset_list_autherror_exits(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
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


def test_dataset_info(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
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


def test_dataset_info_redacts_signed_download_url_by_default(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        return_value={
            "id": "ds-1",
            "download_url": "https://signed.example/file?token=secret",
        },
    ):
        run_cli(["dataset", "info", "ds-1"])

    out = capsys.readouterr().out
    assert "download_url: <redacted>" in out
    assert "token=secret" not in out


def test_dataset_info_apierror_exits(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch(
        "dagnam._core.client.DagnamClient.get_dataset_meta",
        side_effect=APIError(500, "boom"),
    ):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "info", "ds-1"])


def test_dataset_download(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam.load_dataset", return_value=None):
        run_cli(["dataset", "download", "ds-1"])
    assert "downloaded" in capsys.readouterr().out


def test_dataset_download_passes_output_dir_to_loader(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    output_dir = tmp_path / "downloads"
    with mock.patch("dagnam.load_dataset", return_value=None) as load_dataset:
        run_cli(["dataset", "download", "ds-1", "--output-dir", str(output_dir)])

    load_dataset.assert_called_once_with("ds-1", cache_dir=str(output_dir), show_progress=False)
    assert str(output_dir / "ds-1") in capsys.readouterr().out


def test_dataset_download_no_progress_passes_loader_flag(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    output_dir = tmp_path / "downloads"
    with mock.patch("dagnam.load_dataset", return_value=None) as load_dataset:
        run_cli(["dataset", "download", "ds-1", "--output-dir", str(output_dir), "--no-progress"])

    load_dataset.assert_called_once_with("ds-1", cache_dir=str(output_dir), show_progress=False)


def test_dataset_download_apierror_exits(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.load_dataset", side_effect=APIError(500, "boom")):
        with pytest.raises(SystemExit):
            run_cli(["dataset", "download", "ds-1"])


# ---------------------------------------------------------------- cache


def test_cache_clear(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    (cache_dir / "ds-1").mkdir()
    (cache_dir / "ds-1" / "data").write_text("x")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)
    run_cli(["cache", "clear"])
    out = capsys.readouterr().out
    assert "Cleared" in out or "cleared" in out.lower()


def test_cache_clear_dry_run_keeps_cache(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    entry = cache_dir / "ds-1"
    entry.mkdir(parents=True)
    (entry / "data").write_text("x")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)

    run_cli(["cache", "clear", "--dry-run"])

    assert entry.exists()
    assert "Would clear" in capsys.readouterr().out


def test_cache_clear_dataset_id_only_removes_target(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    (cache_dir / "ds-1").mkdir(parents=True)
    (cache_dir / "ds-1" / "data").write_text("x")
    (cache_dir / "ds-2").mkdir()
    (cache_dir / "ds-2" / "data").write_text("y")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)

    run_cli(["cache", "clear", "--dataset-id", "ds-1"])

    assert not (cache_dir / "ds-1").exists()
    assert (cache_dir / "ds-2").exists()
    assert "ds-1" in capsys.readouterr().out


def test_cache_list_with_entries(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
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


def test_inference_batch_loads_from_file(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    fp = tmp_path / "data.json"
    fp.write_text("[1, 2, 3]")
    with mock.patch("dagnam.inference_batch", return_value=[1, 2, 3]):
        run_cli(["inference", "batch", "dep-1", "--inputs", f"@{fp}"])
    assert "1" in capsys.readouterr().out


def test_inference_accepts_explicit_file_flags(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    input_path = tmp_path / "input.json"
    inputs_path = tmp_path / "inputs.json"
    input_path.write_text('{"input":"hello"}', encoding="utf-8")
    inputs_path.write_text('[{"input":"hello"}]', encoding="utf-8")

    with mock.patch("dagnam.inference", return_value={"ok": True}) as infer, mock.patch(
        "dagnam.inference_batch", return_value=[{"ok": True}]
    ) as infer_batch:
        run_cli(["inference", "run", "dep-1", "--input-file", str(input_path)])
        run_cli(["inference", "batch", "dep-1", "--inputs-file", str(inputs_path)])

    infer.assert_called_once_with("dep-1", {"input": "hello"})
    infer_batch.assert_called_once_with("dep-1", [{"input": "hello"}])
    assert "ok" in capsys.readouterr().out


def test_inference_json_file_flags_accept_utf8_bom(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b'\xef\xbb\xbf{"input":"hello"}')

    with mock.patch("dagnam.inference", return_value={"ok": True}) as infer:
        run_cli(["inference", "run", "dep-1", "--input-file", str(input_path)])

    infer.assert_called_once_with("dep-1", {"input": "hello"})
    assert "ok" in capsys.readouterr().out


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
    deployments = SimpleNamespace(health=mock.Mock(return_value={"status": "healthy"}))
    with mock.patch("dagnam.deployments", deployments):
        run_cli(["inference", "health", "dep-1"])
    deployments.health.assert_called_once_with("dep-1")
    assert json.loads(capsys.readouterr().out) == {"status": "healthy"}


def test_inference_health_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "health.json"
    deployments = SimpleNamespace(health=mock.Mock(return_value={"status": "healthy"}))
    with mock.patch("dagnam.deployments", deployments):
        run_cli(["inference", "health", "dep-1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "healthy"}


def test_inference_health_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    deployments = SimpleNamespace(health=mock.Mock(side_effect=APIError(500, "boom")))
    with mock.patch("dagnam.deployments", deployments):
        with pytest.raises(SystemExit):
            run_cli(["inference", "health", "dep-1"])


# ---------------------------------------------------------------- checkpoint


def test_checkpoint_list_empty(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_API_KEY", "k")
    with mock.patch("dagnam._core.client.DagnamClient.list_checkpoints", return_value=[]):
        run_cli(["checkpoint", "list", "job-1"])
    assert "No checkpoints" in capsys.readouterr().out


def test_checkpoint_list_with_rows(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
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


def test_checkpoint_download(
    run_cli: CliRunner, capsys: StrCapture, monkeypatch: PytestMonkeyPatch
) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path"):
        run_cli(["checkpoint", "download", "job-1"])
    assert "/some/path" in capsys.readouterr().out


def test_checkpoint_download_omits_checkpoint_id_for_latest(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path") as download:
        run_cli(["checkpoint", "download", "job-1"])
    download.assert_called_once_with("job-1", None)


def test_checkpoint_download_requests_best_checkpoint(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value="/some/path") as download:
        run_cli(["checkpoint", "download", "job-1", "best"])
    download.assert_called_once_with("job-1", None, prefer_best=True)


def test_checkpoint_download_passes_output_dir(run_cli: CliRunner, tmp_path: Path) -> None:
    with mock.patch("dagnam.download_checkpoint", return_value=tmp_path / "cp.pt") as download:
        run_cli(["checkpoint", "download", "job-1", "cp-1", "--output-dir", str(tmp_path)])

    download.assert_called_once_with("job-1", "cp-1", cache_dir=tmp_path)


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


# ---------------------------------------------------------------- hub


def test_hub_search(run_cli: CliRunner, capsys: StrCapture) -> None:
    model_id = "model-123456-7890"
    fake = SimpleNamespace(
        search=mock.Mock(
            return_value={
                "items": [
                    {
                        "id": model_id,
                        "name": "Tiny ViT",
                        "framework": "pytorch",
                        "task_type": "classification",
                    }
                ],
                "total": 1,
            }
        )
    )
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "search", "--search", "vit"])
    out = capsys.readouterr().out
    assert model_id in out
    assert "Tiny ViT" in out
    assert "classification" in out
    assert '"items"' not in out


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
        featured=mock.Mock(return_value=[{"id": "a", "name": "Featured"}]),
        trending=mock.Mock(return_value=[{"id": "t", "name": "Trending"}]),
    )
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "star", "m1"])
        run_cli(["hub", "unstar", "m1"])
        run_cli(["hub", "fork", "m1"])
        run_cli(["hub", "featured"])
        run_cli(["hub", "trending", "--days", "14"])
    capsys.readouterr()


def test_hub_featured_output_saves_full_json(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    output = tmp_path / "featured.json"
    payload = [{"id": "a", "name": "Featured"}]
    fake = SimpleNamespace(featured=mock.Mock(return_value=payload))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "featured", "--output", str(output)])
    out = capsys.readouterr().out
    assert "Featured" in out
    assert '"name"' not in out
    assert json.loads(output.read_text(encoding="utf-8")) == payload


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


def test_codegen_generate_writes_json_output(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    output = tmp_path / "generated.json"
    fake = SimpleNamespace(
        generate=mock.Mock(return_value={"files": [{"name": "model.py", "content": "x"}]}),
    )

    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "generate", "p1", "--output", str(output)])

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "files": [{"name": "model.py", "content": "x"}]
    }
    out = capsys.readouterr().out
    assert "Wrote code generation response" in out
    assert "model.py" not in out


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
    assert login_mod.format_ascii_art() in out
    assert "Don't have an API key yet?" in out
    assert "http://localhost:5173" in out


def test_login_apierror_exits(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
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


def test_login_preserves_existing_config(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
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


def test_login_persists_training_metrics_path(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    metrics_path = tmp_path / "metrics" / "events.jsonl"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _good_key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login", "--training-metrics-path", str(metrics_path)])
    data = json.loads(config_file.read_text())
    assert data["api_key"] == "good-key"
    assert data["training_metrics_path"] == str(metrics_path)


def test_login_uses_default_training_metrics_path_when_non_interactive(
    run_cli: CliRunner,
    tmp_path: Path,
    monkeypatch: PytestMonkeyPatch,
    capsys: StrCapture,
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _good_key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login"])
    data = json.loads(config_file.read_text(encoding="utf-8"))
    metrics_path = Path(data["training_metrics_path"])
    assert metrics_path.name == "dagnam_metrics.jsonl"
    assert metrics_path.parent.name == "training-metrics"
    assert "Training metrics path:" in capsys.readouterr().out


def test_login_with_custom_api_url(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    config_dir = tmp_path / ".dagnam"
    config_file = config_dir / "config.json"
    monkeypatch.setattr("dagnam._core.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", config_file)
    monkeypatch.setattr("getpass.getpass", _key_prompt)
    with mock.patch("dagnam._core.client.DagnamClient.list_datasets", return_value=[]):
        run_cli(["login", "--api-url", "https://custom"])
    data = json.loads(config_file.read_text())
    assert data["api_url"] == "https://custom"


def test_login_corrupt_existing_config_starts_fresh(
    run_cli: CliRunner, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
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
    assert data["api_key"] == "k"
    assert Path(data["training_metrics_path"]).name == "dagnam_metrics.jsonl"


# ---------------------------------------------------------------- training jobs


def test_training_create_passes_hyperparameters(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    create = mock.Mock(return_value={"id": "j1", "status": "pending"})
    with mock.patch("dagnam.create_training_job", create):
        run_cli(
            [
                "training", "create", "p1",
                "--epochs", "2",
                "--batch-size", "32",
                "--learning-rate", "0.001",
                "--optimizer", "adam",
                "--loss-function", "cross_entropy",
                "--dataset-id", "ds1",
                "--max-duration-seconds", "600",
                "--confirm-resource-warning",
            ]
        )
    kwargs = create.call_args.kwargs
    assert create.call_args.args == ("p1",)
    assert kwargs["epochs"] == 2
    assert kwargs["batch_size"] == 32
    assert kwargs["learning_rate"] == 0.001
    assert kwargs["optimizer"] == "adam"
    assert kwargs["training_dataset_id"] == "ds1"
    assert kwargs["max_duration_seconds"] == 600
    assert kwargs["confirm_resource_warning"] is True
    assert '"id": "j1"' in capsys.readouterr().out


def test_training_create_parses_config_overrides_file(
    run_cli: CliRunner, tmp_path: Path
) -> None:
    overrides = tmp_path / "cfg.json"
    overrides.write_text('{"logging_config": {"log_frequency": 5}}', encoding="utf-8")
    create = mock.Mock(return_value={"id": "j1"})
    with mock.patch("dagnam.create_training_job", create):
        run_cli(
            [
                "training", "create", "p1",
                "--epochs", "1", "--batch-size", "8", "--learning-rate", "0.01",
                "--optimizer", "sgd", "--loss-function", "mse", "--dataset-id", "ds1",
                "--config", f"@{overrides}",
            ]
        )
    assert create.call_args.kwargs["config_overrides"] == {"logging_config": {"log_frequency": 5}}


def test_training_create_bad_config_exits_cleanly(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(
            [
                "training", "create", "project-1",
                "--epochs", "1",
                "--batch-size", "2",
                "--learning-rate", "0.1",
                "--optimizer", "adam",
                "--loss-function", "mse",
                "--dataset-id", "dataset-1",
                "--config", "{bad-json",
            ]
        )

    assert exc.value.code == 1
    assert "Could not read --config JSON" in capsys.readouterr().err


def test_training_attach_help_documents_replay_only_exit(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["training", "attach", "--help"])
    assert excinfo.value.code == 0
    assert "replay existing events and exit" in capsys.readouterr().out


def test_training_list_concise_and_verbose(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {
        "items": [
            {
                "id": "j1",
                "status": "running",
                "framework": "pytorch",
                "current_epoch": 1,
                "total_epochs": 2,
                "progress_percentage": 50,
                "created_at": "2026-05-26T10:00:00",
            }
        ],
        "total": 1,
    }
    with mock.patch("dagnam.list_training_jobs", mock.Mock(return_value=payload)):
        run_cli(["training", "list"])
        run_cli(["training", "list", "--verbose"])
    out = capsys.readouterr().out
    assert "j1" in out
    assert "running" in out
    assert '"items"' in out  # from the verbose run


def test_training_list_empty(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.list_training_jobs", mock.Mock(return_value={"items": [], "total": 0})):
        run_cli(["training", "list"])
    assert "No training jobs found" in capsys.readouterr().out


def test_training_list_json_writes_output(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    payload = {"items": [{"id": "j1"}], "total": 1}
    output = tmp_path / "training.json"
    with mock.patch("dagnam.list_training_jobs", mock.Mock(return_value=payload)):
        run_cli(["training", "list", "--json", "--output", str(output)])
    assert json.loads(capsys.readouterr().out) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_training_get(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {
        "id": "j1",
        "status": "running",
        "framework": "pytorch",
        "current_epoch": 1,
        "total_epochs": 3,
        "progress_percentage": 33,
    }
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1"])
    out = capsys.readouterr().out
    assert "Training job j1" in out
    assert "Epoch: 1/3" in out
    assert '"id": "j1"' not in out


def test_training_get_json_prints_full_payload(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"id": "j1", "status": "running"}
    with mock.patch("dagnam.get_training_job", mock.Mock(return_value=payload)):
        run_cli(["training", "get", "j1", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_training_logs_dispatches_filters(run_cli: CliRunner, capsys: StrCapture) -> None:
    logs = mock.Mock(return_value={"items": []})
    with mock.patch("dagnam.training_logs", logs):
        run_cli(["training", "logs", "j1", "--log-level", "error", "--limit", "5"])
    logs.assert_called_once_with("j1", log_level="error", source=None, page=1, limit=5)
    assert json.loads(capsys.readouterr().out) == {"items": []}


def test_training_metrics_dispatches_filters(run_cli: CliRunner, capsys: StrCapture) -> None:
    metrics = mock.Mock(return_value={"items": []})
    with mock.patch("dagnam.training_metrics", metrics):
        run_cli(["training", "metrics", "j1", "--metric-type", "train_loss", "--epoch-summary"])
    metrics.assert_called_once_with(
        "j1",
        metric_type="train_loss",
        epoch_start=None,
        epoch_end=None,
        epoch_summary=True,
        page=1,
        limit=100,
    )
    assert json.loads(capsys.readouterr().out) == {"items": []}


def test_training_metrics_summary_dispatches(run_cli: CliRunner, capsys: StrCapture) -> None:
    summary = mock.Mock(return_value={"best_epoch": 2})
    with mock.patch("dagnam.training_metrics_summary", summary):
        run_cli(["training", "metrics-summary", "j1"])
    summary.assert_called_once_with("j1")
    assert json.loads(capsys.readouterr().out) == {"best_epoch": 2}


def test_training_cancel_prints_message(run_cli: CliRunner, capsys: StrCapture) -> None:
    cancel = mock.Mock(return_value={"message": "Training job cancelled successfully"})
    with mock.patch("dagnam.cancel_training_job", cancel):
        run_cli(["training", "cancel", "j1"])
    cancel.assert_called_once_with("j1")
    assert "cancelled successfully" in capsys.readouterr().out


def test_training_delete_bulk(run_cli: CliRunner, capsys: StrCapture) -> None:
    delete = mock.Mock(return_value={"deleted": 2})
    with mock.patch("dagnam.delete_training_jobs", delete):
        run_cli(["training", "delete", "j1", "j2"])
    delete.assert_called_once_with(["j1", "j2"])
    assert '"deleted": 2' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("cmd_args", "attr"),
    [
        (["training", "get", "j1"], "get_training_job"),
        (["training", "cancel", "j1"], "cancel_training_job"),
        (["training", "delete", "j1"], "delete_training_jobs"),
        (["training", "list"], "list_training_jobs"),
    ],
)
def test_training_apierrors_exit(run_cli: CliRunner, cmd_args: list[str], attr: str) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch(f"dagnam.{attr}", mock.Mock(side_effect=APIError(500, "boom"))):
        with pytest.raises(SystemExit):
            run_cli(cmd_args)


# ---------------------------------------------------------------- usage


def test_usage_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {
        "plan": {"code": "pro", "display_name": "Pro"},
        "read_only_grace": False,
        "limits": [
            {"key": "concurrent_training_jobs", "current": 1, "limit": 3},
            {"key": "training_minutes", "current": 10, "limit": None},
        ],
    }
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage"])
    out = capsys.readouterr().out
    assert "Plan: Pro" in out
    assert "concurrent_training_jobs" in out
    assert "training_minutes" in out


def test_usage_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    snapshot = {"plan": {"code": "free"}, "limits": []}
    fake = SimpleNamespace(entitlements=mock.Mock(return_value=snapshot))
    with mock.patch("dagnam.account", fake):
        run_cli(["usage", "--json"])
    assert json.loads(capsys.readouterr().out) == snapshot


def test_usage_apierror_exits(run_cli: CliRunner) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(entitlements=mock.Mock(side_effect=APIError(500, "boom")))
    with mock.patch("dagnam.account", fake):
        with pytest.raises(SystemExit):
            run_cli(["usage"])


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
                "projects", "architecture", "p1",
                "--diagram", f"@{diagram}",
                "--config", f"@{config}",
                "--message", "init",
            ]
        )
    save.assert_called_once_with(
        "p1", {"nodes": []}, {"layers": []}, commit_message="init"
    )
    assert '"version_id": "v1"' in capsys.readouterr().out


def test_projects_architecture_accepts_json_literals(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    save = mock.Mock(return_value={"version_id": "v1"})
    fake = SimpleNamespace(save_architecture=save)
    with mock.patch("dagnam.projects", fake):
        run_cli(
            [
                "projects", "architecture", "p1",
                "--diagram", '{"nodes": [1]}',
                "--config", '{"layers": [2]}',
            ]
        )
    save.assert_called_once_with(
        "p1", {"nodes": [1]}, {"layers": [2]}, commit_message=None
    )
    capsys.readouterr()
