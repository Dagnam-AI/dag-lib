"""CLI codegen subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


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
        run_cli(["codegen", "download", "p1", "--framework", "pytorch", "--dest", "/tmp/out"])
    out = capsys.readouterr().out
    assert "Generation started (task t1)." in out
    assert "/tmp/code.zip" in out
    fake.download.assert_called_once()
    assert fake.download.call_args.kwargs["dest"] == "/tmp/out"


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
    # Human stdout is concise; full file content is not dumped to stdout.
    assert "Generated 1 file(s)." in out
    assert "model.py" not in out


def test_codegen_generate_json_stdout(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(generate=mock.Mock(return_value={"files": [{"name": "model.py"}]}))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "generate", "p1", "--json"])
    out = capsys.readouterr().out
    assert json.loads(out) == {"files": [{"name": "model.py"}]}


def test_render_generate_variants(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(generate=mock.Mock(return_value="unexpected"))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "generate", "p1"])
    assert "Code generated." in capsys.readouterr().out


def test_render_preview_with_files(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        preview=mock.Mock(return_value={"files": [{"name": "a.py"}, {"name": "b.py"}]})
    )
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "preview", "p1"])
    out = capsys.readouterr().out
    assert "Preview: 2 file(s): a.py, b.py" in out


def test_render_preview_ready(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(preview=mock.Mock(return_value={"code": "..."}))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "preview", "p1"])
    assert "Preview ready." in capsys.readouterr().out


def test_render_validate_valid(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(validate=mock.Mock(return_value={"valid": True}))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "validate", "p1"])
    assert "Project is valid." in capsys.readouterr().out


def test_render_validate_invalid_with_errors(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(
        validate=mock.Mock(return_value={"valid": False, "errors": ["bad shape", "no output"]})
    )
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "validate", "p1"])
    out = capsys.readouterr().out
    assert "Project is invalid: bad shape; no output" in out


def test_render_validate_non_dict(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(validate=mock.Mock(return_value=["not", "a", "dict"]))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "validate", "p1"])
    assert "Project is invalid." in capsys.readouterr().out


def test_render_preview_non_dict(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(preview=mock.Mock(return_value=None))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "preview", "p1"])
    assert "Preview ready." in capsys.readouterr().out


def test_render_generate_non_dict(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(generate=mock.Mock(return_value=None))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "generate", "p1"])
    assert "Code generated." in capsys.readouterr().out


def test_codegen_download_no_dest(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(download=mock.Mock(return_value="/tmp/code.zip"))
    with mock.patch("dagnam.codegen", fake):
        run_cli(["codegen", "download", "p1"])
    assert fake.download.call_args.kwargs["dest"] is None
    assert "/tmp/code.zip" in capsys.readouterr().out


def test_codegen_download_no_output_flag(run_cli: CliRunner) -> None:
    """The download subcommand is a clean break: --output no longer exists."""
    fake = SimpleNamespace(download=mock.Mock(return_value="/tmp/code.zip"))
    with mock.patch("dagnam.codegen", fake):
        with pytest.raises(SystemExit):
            run_cli(["codegen", "download", "p1", "--output", "/tmp/out"])


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
