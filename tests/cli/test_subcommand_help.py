"""CLI subcommand-help behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dagnam.cli import main as cli_main

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


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
