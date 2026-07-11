"""CLI inference subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


# ---------------------------------------------------------------- inference


def test_inference_run(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.inference", return_value={"label": "ok"}):
        run_cli(["inference", "run", "dep-1", "--input", '{"x":1}'])
    assert json.loads(capsys.readouterr().out) == {"label": "ok"}


def test_inference_run_bad_json(run_cli: CliRunner, monkeypatch: PytestMonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run_cli(["inference", "run", "dep-1", "--input", "not-json"])


def test_inference_run_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.inference", side_effect=APIError(500, "boom")):
        assert run_cli(["inference", "run", "dep-1", "--input", '{"x":1}']) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


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


def test_inference_batch_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.inference_batch", side_effect=APIError(500, "boom")):
        assert run_cli(["inference", "batch", "dep-1", "--inputs", "[1,2]"]) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


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

    with (
        mock.patch("dagnam.inference", return_value={"ok": True}) as infer,
        mock.patch("dagnam.inference_batch", return_value=[{"ok": True}]) as infer_batch,
    ):
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


def test_inference_health_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    deployments = SimpleNamespace(health=mock.Mock(side_effect=APIError(500, "boom")))
    with mock.patch("dagnam.deployments", deployments):
        assert run_cli(["inference", "health", "dep-1"]) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


def test_inference_run_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    with mock.patch("dagnam.inference", return_value={"label": "ok"}):
        run_cli(["inference", "run", "dep-1", "--input", '{"x":1}', "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {"label": "ok"}


def test_inference_batch_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    with mock.patch("dagnam.inference_batch", return_value=[{"y": 1}]):
        run_cli(["inference", "batch", "dep-1", "--inputs", "[1]", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == [{"y": 1}]


def test_inference_schema(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch(
        "dagnam.inference_schema",
        return_value={"input_schema": {"type": "object"}, "output_schema": {}},
    ):
        run_cli(["inference", "schema", "dep-1"])
    assert "input_schema" in capsys.readouterr().out


def test_inference_schema_writes_output_file(run_cli: CliRunner, tmp_path: Path) -> None:
    output = tmp_path / "schema.json"
    with mock.patch(
        "dagnam.inference_schema", return_value={"input_schema": {}, "output_schema": {}}
    ):
        run_cli(["inference", "schema", "dep-1", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "input_schema": {},
        "output_schema": {},
    }


def test_inference_schema_apierror_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import APIError

    with mock.patch("dagnam.inference_schema", side_effect=APIError(500, "boom")):
        assert run_cli(["inference", "schema", "dep-1"]) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


# ---------------------------------------------------------------- stream


def _fake_events(*events):
    from dagnam._core.sse import SSEEvent

    return [SSEEvent(event=e, data=d) for e, d in events]


def test_inference_stream_prints_tokens_incrementally(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    events = _fake_events(
        ("token", {"token": "he", "index": 1}),
        ("token", {"token": "llo", "index": 2}),
        ("complete", {"done": True, "total_tokens": 2, "latency_ms": 12.5}),
    )
    with mock.patch("dagnam.inference_stream", return_value=iter(events)) as m:
        assert run_cli(["inference", "stream", "dep-1", "--input", '{"text":"hi"}']) == 0
    m.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == "hello\n"
    assert "2 tokens" in captured.err


def test_inference_stream_json_emits_ndjson(run_cli: CliRunner, capsys: StrCapture) -> None:
    events = _fake_events(("token", {"token": "a", "index": 1}), ("complete", {"done": True}))
    with mock.patch("dagnam.inference_stream", return_value=iter(events)):
        assert run_cli(["inference", "stream", "dep-1", "--input", "{}", "--json"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"event": "token", "data": {"token": "a", "index": 1}},
        {"event": "complete", "data": {"done": True}},
    ]


def test_inference_stream_error_event_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    events = _fake_events(("token", {"token": "a"}), ("error", {"message": "model blew up"}))
    with mock.patch("dagnam.inference_stream", return_value=iter(events)):
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["inference", "stream", "dep-1", "--input", "{}"])
    assert exc_info.value.code == 1
    assert "model blew up" in capsys.readouterr().err


def test_inference_stream_json_error_event_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    events = _fake_events(("token", {"token": "a"}), ("error", {"message": "kaboom"}))
    with mock.patch("dagnam.inference_stream", return_value=iter(events)):
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["inference", "stream", "dep-1", "--input", "{}", "--json"])
    assert exc_info.value.code == 1
    assert "kaboom" in capsys.readouterr().err


def test_inference_stream_disconnect_exits_1(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam._core.exceptions import StreamError

    def _gen():
        from dagnam._core.sse import SSEEvent

        yield SSEEvent(event="token", data={"token": "a"})
        raise StreamError("Inference stream for dep-1 dropped mid-stream: boom")

    with mock.patch("dagnam.inference_stream", return_value=_gen()):
        assert run_cli(["inference", "stream", "dep-1", "--input", "{}"]) == 1
    assert "dropped mid-stream" in capsys.readouterr().err


def test_inference_stream_skips_non_token_and_unknown_events(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    events = _fake_events(
        ("token", {"index": 1}),  # no "token" key -> nothing printed
        ("meta", {"info": "x"}),  # unknown event -> ignored
        ("complete", {"done": True}),  # completes with zero tokens printed
    )
    with mock.patch("dagnam.inference_stream", return_value=iter(events)):
        assert run_cli(["inference", "stream", "dep-1", "--input", "{}"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "0 tokens" in captured.err


def test_inference_stream_error_first_event_exits(run_cli: CliRunner, capsys: StrCapture) -> None:
    events = _fake_events(("error", {"message": "immediate failure"}))
    with mock.patch("dagnam.inference_stream", return_value=iter(events)):
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["inference", "stream", "dep-1", "--input", "{}"])
    assert exc_info.value.code == 1
    assert "immediate failure" in capsys.readouterr().err


def test_inference_stream_reads_input_file(
    run_cli: CliRunner, tmp_path: Path, capsys: StrCapture
) -> None:
    input_path = tmp_path / "in.json"
    input_path.write_text('{"text": "hi"}', encoding="utf-8")
    events = _fake_events(("complete", {"done": True}))
    with mock.patch("dagnam.inference_stream", return_value=iter(events)) as m:
        assert run_cli(["inference", "stream", "dep-1", "--input-file", str(input_path)]) == 0
    m.assert_called_once_with("dep-1", {"text": "hi"})


def test_inference_stream_bad_json_input_exits(run_cli: CliRunner) -> None:
    with pytest.raises(SystemExit):
        run_cli(["inference", "stream", "dep-1", "--input", "not-json"])
