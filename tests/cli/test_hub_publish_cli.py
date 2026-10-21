"""CLI ``hub publish``, ``hub version create`` and ``hub derivatives``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture

DONE: dict[str, object] = {"model": {"id": "m1"}, "files": [], "version": None}
BASE = [
    "hub",
    "publish",
    "--name",
    "n",
    "--description",
    "d",
    "--task-type",
    "other",
    "--framework",
    "pytorch",
]
UUID = "11111111-2222-3333-4444-555555555555"


def _design(tmp_path: Path) -> str:
    path = tmp_path / "design.json"
    path.write_text(json.dumps({"diagram_state": {}, "architecture_config": {}}), encoding="utf-8")
    return f"@{path}"


def test_publish_a_design_file_defaults_to_the_design_stage(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    with mock.patch("dagnam.hub.publish", return_value=DONE) as m:
        assert run_cli([*BASE, "--architecture", _design(tmp_path)]) == 0
    kwargs = m.call_args.kwargs
    assert kwargs["files"] == []
    assert kwargs["architecture"] == {"diagram_state": {}, "architecture_config": {}}
    assert kwargs["stage"] == "design"
    assert kwargs["project_id"] is None
    assert kwargs["registry_version_id"] is None
    assert kwargs["license"] == "mit"
    assert kwargs["version"] == "1.0.0"
    assert json.loads(capsys.readouterr().out)["model"]["id"] == "m1"


def test_publish_a_project_design_with_its_version(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.hub.publish", return_value=DONE) as m:
        assert run_cli([*BASE, "--project", "P", "--architecture-version", "V"]) == 0
    kwargs = m.call_args.kwargs
    assert (kwargs["project_id"], kwargs["architecture_version_id"]) == ("P", "V")
    assert kwargs["architecture"] is None
    assert kwargs["stage"] == "design"


def test_architecture_version_needs_a_project(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.publish") as m, pytest.raises(SystemExit) as exc_info:
        run_cli([*BASE, "--architecture-version", "V"])
    assert exc_info.value.code == 2
    assert "--architecture-version needs --project" in capsys.readouterr().err
    m.assert_not_called()


def test_post_trained_publish_passes_stage_and_base(run_cli: CliRunner, tmp_path: Path) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    argv = [*BASE, "--stage", "fine-tuned", "--base", "Qwen/Qwen2.5-0.5B-Instruct"]
    with mock.patch("dagnam.hub.publish", return_value=DONE) as m:
        assert run_cli([*argv, "--file", str(weights)]) == 0
    kwargs = m.call_args.kwargs
    assert kwargs["stage"] == "fine-tuned"
    assert kwargs["base"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert kwargs["files"] == [str(weights)]


def test_a_file_needs_a_stage(run_cli: CliRunner, capsys: StrCapture, tmp_path: Path) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    with mock.patch("dagnam.hub.publish") as m, pytest.raises(SystemExit) as exc_info:
        run_cli([*BASE, "--file", str(weights)])
    assert exc_info.value.code == 2
    assert "--stage is required with --file" in capsys.readouterr().err
    m.assert_not_called()


@pytest.mark.parametrize("extra", [[], ["--stage", "design"]])
def test_a_design_needs_a_source_before_any_call(
    run_cli: CliRunner, capsys: StrCapture, extra: list[str]
) -> None:
    with mock.patch("dagnam.hub.publish") as m, pytest.raises(SystemExit) as exc_info:
        run_cli([*BASE, *extra])
    assert exc_info.value.code == 2
    assert "a design needs --architecture or --project" in capsys.readouterr().err
    m.assert_not_called()


def test_a_registry_version_publishes_a_card_without_a_stage(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.hub.publish", return_value=DONE) as m:
        assert run_cli([*BASE, "--registry-version", "R"]) == 0
    kwargs = m.call_args.kwargs
    assert kwargs["registry_version_id"] == "R"
    assert kwargs["stage"] is None


def test_licence_metadata_and_first_version_pass_through(
    run_cli: CliRunner, tmp_path: Path
) -> None:
    card = tmp_path / "card.json"
    card.write_text('{"intended_use": "x"}', encoding="utf-8")
    argv = [*BASE, "--registry-version", "R", "--license", "custom", "--license-name", "Acme"]
    argv += ["--license-url", "https://example.com/l", "--metadata", f"@{card}"]
    argv += ["--version", "2.0.0", "--changelog", "c", "--tags", "a, b", "--model-id", "d1"]
    with mock.patch("dagnam.hub.publish", return_value=DONE) as m:
        assert run_cli(argv) == 0
    kwargs = m.call_args.kwargs
    assert kwargs["license"] == "custom"
    assert kwargs["license_name"] == "Acme"
    assert kwargs["license_url"] == "https://example.com/l"
    assert kwargs["metadata"] == {"intended_use": "x"}
    assert (kwargs["version"], kwargs["changelog"]) == ("2.0.0", "c")
    assert kwargs["tags"] == ["a", "b"]
    assert kwargs["model_id"] == "d1"


def test_publish_prints_file_progress_to_stderr(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")

    def fake_publish(**kwargs: object) -> dict[str, object]:
        progress = kwargs["on_file_progress"]
        assert callable(progress)
        progress(str(weights), 1, 1, "uploaded")
        return DONE

    with mock.patch("dagnam.hub.publish", side_effect=fake_publish):
        assert run_cli([*BASE, "--stage", "adapter", "--file", str(weights)]) == 0
    assert "[1/1] w.safetensors: uploaded" in capsys.readouterr().err


def test_publish_without_a_model_id_in_the_result_still_prints_a_next_step(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    with mock.patch("dagnam.hub.publish", return_value={"model": None}):
        assert run_cli([*BASE, "--architecture", _design(tmp_path)]) == 0
    assert "dagnam hub get <model-id>" in capsys.readouterr().err


def test_version_create_publishes_a_version(run_cli: CliRunner, tmp_path: Path) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    argv = ["hub", "version", "create", "m1", "--version", "1.1.0", "--file", str(weights)]
    with mock.patch("dagnam.hub.publish_version", return_value={"id": "v2"}) as m:
        assert run_cli([*argv, "--changelog", "c"]) == 0
    m.assert_called_once_with(
        "m1",
        "1.1.0",
        files=[str(weights)],
        changelog="c",
        architecture=None,
        project_id=None,
        architecture_version_id=None,
        on_file_progress=mock.ANY,
    )


def test_version_create_with_a_design_and_progress(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    def fake(*_args: object, **kwargs: object) -> dict[str, object]:
        progress = kwargs["on_file_progress"]
        assert callable(progress)
        progress("a/w.safetensors", 1, 1, "uploading")
        return {"id": "v2"}

    argv = ["hub", "version", "create", "m1", "--version", "1.1.0"]
    argv += ["--architecture", _design(tmp_path), "--project", "P", "--architecture-version", "V"]
    with mock.patch("dagnam.hub.publish_version", side_effect=fake) as m:
        assert run_cli(argv) == 0
    kwargs = m.call_args.kwargs
    assert kwargs["architecture"] == {"diagram_state": {}, "architecture_config": {}}
    assert (kwargs["project_id"], kwargs["architecture_version_id"]) == ("P", "V")
    captured = capsys.readouterr()
    assert "[1/1] w.safetensors: uploading" in captured.err
    assert json.loads(captured.out) == {"id": "v2"}


def test_version_create_architecture_version_needs_a_project(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    argv = ["hub", "version", "create", "m1", "--version", "1.1.0", "--architecture-version", "V"]
    with mock.patch("dagnam.hub.publish_version") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(argv)
    assert exc_info.value.code == 2
    assert "--architecture-version needs --project" in capsys.readouterr().err
    m.assert_not_called()


def test_derivatives_renders_a_table(run_cli: CliRunner, capsys: StrCapture) -> None:
    result = {"items": [{"id": UUID, "name": "Tiny", "stage": "adapter", "base_model_name": "B"}]}
    fake = SimpleNamespace(derivatives=mock.Mock(return_value=result))
    with mock.patch("dagnam.hub", fake):
        assert run_cli(["hub", "derivatives", "m1", "--stage", "adapter"]) == 0
    fake.derivatives.assert_called_once_with("m1", stage=["adapter"], page=1, limit=20)
    out = capsys.readouterr().out
    assert "Tiny" in out
    assert "adapter" in out


def test_derivatives_without_a_stage_filter(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(derivatives=mock.Mock(return_value={"items": []}))
    with mock.patch("dagnam.hub", fake):
        assert run_cli(["hub", "derivatives", "m1", "--page", "2", "--limit", "5"]) == 0
    fake.derivatives.assert_called_once_with("m1", stage=None, page=2, limit=5)
    assert "No hub models found." in capsys.readouterr().out
