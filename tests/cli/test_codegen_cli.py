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
