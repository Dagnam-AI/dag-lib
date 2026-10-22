"""CLI hub subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


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
        (["hub", "upload-file", "m1", "/tmp/w.bin"], "upload_file"),
        (["hub", "download", "m1"], "download"),
    ],
)
def test_hub_apierrors_exit(
    run_cli: CliRunner, capsys: StrCapture, cmd_args: list[str], attr: str
) -> None:
    from dagnam._core.exceptions import APIError

    fake = SimpleNamespace(**{attr: mock.Mock(side_effect=APIError(500, "boom"))})
    with mock.patch("dagnam.hub", fake):
        assert run_cli(cmd_args) == 1
    err = capsys.readouterr().err
    assert "Error: the Dagnam API had an internal error (HTTP 500)" in err
    assert "boom" in err


def test_hub_search_empty_message(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(search=mock.Mock(return_value={"items": [], "total": 0}))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "search"])
    assert "No hub models found." in capsys.readouterr().out


def test_hub_featured_list_with_non_dict_item(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(featured=mock.Mock(return_value=["model-name-string"]))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "featured"])
    assert "model-name-string" in capsys.readouterr().out


def test_hub_upload_file(run_cli: CliRunner, capsys: StrCapture, tmp_path: Path) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    fake = SimpleNamespace(
        upload_file=mock.Mock(return_value={"id": "f1", "file_name": "weights.bin"})
    )
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "upload-file", "m1", str(f)])
    assert "f1" in capsys.readouterr().out
    fake.upload_file.assert_called_once_with("m1", str(f), version_id=None)


def test_hub_upload_file_into_a_draft_version(run_cli: CliRunner, tmp_path: Path) -> None:
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\x00")
    fake = SimpleNamespace(upload_file=mock.Mock(return_value={"id": "f1"}))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "upload-file", "m1", str(f), "--version-id", "v1"])
    fake.upload_file.assert_called_once_with("m1", str(f), version_id="v1")


def test_hub_download_prints_each_saved_path(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    saved = [str(tmp_path / "model.safetensors"), str(tmp_path / "head.onnx")]
    with mock.patch("dagnam.hub.download", return_value=saved) as m:
        assert run_cli(["hub", "download", "m1", "--out", str(tmp_path)]) == 0
    m.assert_called_once_with("m1", str(tmp_path), file_id=None)
    assert capsys.readouterr().out.splitlines() == saved


def test_hub_download_one_file_into_cwd_by_default(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.download", return_value=["model.safetensors"]) as m:
        assert run_cli(["hub", "download", "m1", "--file-id", "f1"]) == 0
    m.assert_called_once_with("m1", ".", file_id="f1")
    assert capsys.readouterr().out == "model.safetensors\n"


# ---------------------------------------------------------------- publish surface


def test_hub_publish_calls_orchestrator_and_prints_progress(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"x")

    def fake_publish(**kwargs: object) -> dict[str, object]:
        cb = kwargs["on_file_progress"]
        assert callable(cb)
        cb(str(weights), 1, 1, "uploading")
        cb(str(weights), 1, 1, "uploaded")
        return {"model": {"id": "m1"}, "files": [{"id": "f1"}], "version": None, "finalized": True}

    with mock.patch("dagnam.hub.publish", side_effect=fake_publish) as m:
        assert (
            run_cli(
                [
                    "hub",
                    "publish",
                    "--name",
                    "n",
                    "--description",
                    "d",
                    "--task-type",
                    "t",
                    "--framework",
                    "pytorch",
                    "--stage",
                    "adapter",
                    "--file",
                    str(weights),
                ]
            )
            == 0
        )
    kwargs = m.call_args.kwargs
    assert kwargs["name"] == "n"
    assert kwargs["files"] == [str(weights)]
    assert kwargs["visibility"] == "public"
    assert kwargs["model_id"] is None
    captured = capsys.readouterr()
    assert "[1/1]" in captured.err
    assert "weights.bin" in captured.err
    assert json.loads(captured.out)["model"]["id"] == "m1"


def test_hub_publish_upload_failure_exits_1(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    from dagnam._core.exceptions import UploadError

    weights = tmp_path / "w.bin"
    weights.write_bytes(b"x")
    with mock.patch("dagnam.hub.publish", side_effect=UploadError("m1 halted: w.bin failed")):
        assert (
            run_cli(
                [
                    "hub",
                    "publish",
                    "--name",
                    "n",
                    "--description",
                    "d",
                    "--task-type",
                    "t",
                    "--framework",
                    "pytorch",
                    "--stage",
                    "adapter",
                    "--file",
                    str(weights),
                ]
            )
            == 1
        )
    assert "halted" in capsys.readouterr().err


def test_hub_publish_model_id_resumes_a_draft(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    done = {"model": {"id": "d1"}, "files": [], "version": None, "finalized": True}
    with mock.patch("dagnam.hub.publish", return_value=done) as m:
        assert (
            run_cli(
                [
                    "hub",
                    "publish",
                    "--model-id",
                    "d1",
                    "--name",
                    "n",
                    "--description",
                    "d",
                    "--task-type",
                    "classification",
                    "--framework",
                    "pytorch",
                    "--stage",
                    "adapter",
                    "--file",
                    str(weights),
                ]
            )
            == 0
        )
    assert m.call_args.kwargs["model_id"] == "d1"
    assert json.loads(capsys.readouterr().out)["model"]["id"] == "d1"


def test_hub_publish_json_flag_is_accepted_and_prints_the_same_json(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    """publish always prints JSON; --json is accepted so callers that pass it keep working."""
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"x")
    done = {"model": {"id": "m1"}, "files": [], "version": None, "finalized": True}
    argv = [
        "hub",
        "publish",
        "--name",
        "n",
        "--description",
        "d",
        "--task-type",
        "classification",
        "--framework",
        "pytorch",
        "--stage",
        "adapter",
        "--file",
        str(weights),
    ]
    with mock.patch("dagnam.hub.publish", return_value=done):
        assert run_cli(argv) == 0
        plain = capsys.readouterr().out
        assert run_cli([*argv, "--json"]) == 0
        flagged = capsys.readouterr().out
    assert json.loads(plain) == done
    assert flagged == plain


def test_hub_update(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.update", return_value={"id": "m1", "name": "new"}) as m:
        assert run_cli(["hub", "update", "m1", "--name", "new", "--tags", "a,b"]) == 0
    m.assert_called_once_with("m1", name="new", tags=["a", "b"])


def test_hub_update_description_and_visibility(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.hub.update", return_value={"id": "m1"}) as m:
        assert (
            run_cli(["hub", "update", "m1", "--description", "d2", "--visibility", "private"]) == 0
        )
    m.assert_called_once_with("m1", description="d2", visibility="private")


def test_hub_update_lineage_and_licence_fields(run_cli: CliRunner) -> None:
    uuid = "11111111-2222-3333-4444-555555555555"
    argv = [
        "hub",
        "update",
        "m1",
        "--license",
        "gemma",
        "--license-name",
        "N",
        "--license-url",
        "U",
    ]
    with mock.patch("dagnam.hub.update", return_value={"id": "m1"}) as m:
        assert run_cli([*argv, "--base", "Qwen/Qwen2.5"]) == 0
        assert run_cli([*argv, "--base", uuid]) == 0
    first, second = m.call_args_list
    assert first.kwargs == {
        "license": "gemma",
        "license_name": "N",
        "license_url": "U",
        "base_model_name": "Qwen/Qwen2.5",
    }
    assert second.kwargs["base_model_id"] == uuid
    assert "base_model_name" not in second.kwargs


def test_hub_search_filters_by_stage(run_cli: CliRunner) -> None:
    fake = SimpleNamespace(search=mock.Mock(return_value={"items": []}))
    with mock.patch("dagnam.hub", fake):
        run_cli(["hub", "search", "--stage", "design", "--stage", "adapter"])
        run_cli(["hub", "search"])
    first, second = fake.search.call_args_list
    assert first.kwargs["stage"] == ["design", "adapter"]
    assert second.kwargs["stage"] is None


def test_hub_update_requires_a_field(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.update") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(["hub", "update", "m1"])
    assert exc_info.value.code == 1
    m.assert_not_called()


def test_hub_delete_typed_confirmation(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch, capsys: StrCapture
) -> None:
    monkeypatch.setattr("builtins.input", lambda *a: "m1")
    with mock.patch("dagnam.hub.delete", return_value=None) as m:
        assert run_cli(["hub", "delete", "m1"]) == 0
    m.assert_called_once_with("m1")
    assert "deleted" in capsys.readouterr().out.lower()


def test_hub_delete_wrong_confirmation_aborts(
    run_cli: CliRunner, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda *a: "other")
    with mock.patch("dagnam.hub.delete") as m, pytest.raises(SystemExit) as exc_info:
        run_cli(["hub", "delete", "m1"])
    assert exc_info.value.code == 1
    m.assert_not_called()


def test_hub_delete_yes_skips_prompt(run_cli: CliRunner) -> None:
    with mock.patch("dagnam.hub.delete", return_value=None) as m:
        assert run_cli(["hub", "delete", "m1", "--yes"]) == 0
    m.assert_called_once()


def test_hub_versions(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.list_versions", return_value=[{"version": "1.0.0"}]) as m:
        assert run_cli(["hub", "versions", "m1"]) == 0
    m.assert_called_once_with("m1")
    assert json.loads(capsys.readouterr().out)[0]["version"] == "1.0.0"


def test_hub_review(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.add_review", return_value={"id": "r1"}) as m:
        assert run_cli(["hub", "review", "m1", "--rating", "5", "--text", "great"]) == 0
    m.assert_called_once_with("m1", 5, review_text="great")


def test_hub_review_rating_out_of_range_is_usage_error(run_cli: CliRunner) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["hub", "review", "m1", "--rating", "9"])
    assert exc_info.value.code == 2


def test_hub_starred(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.starred", return_value={"items": [{"id": "m1"}]}) as m:
        assert run_cli(["hub", "starred", "--json"]) == 0
    m.assert_called_once_with(sort_by="date_starred", page=1, limit=20)


def test_hub_use_in_studio(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch("dagnam.hub.use_in_studio", return_value={"project_id": "p1"}) as m:
        assert run_cli(["hub", "use-in-studio", "m1"]) == 0
    m.assert_called_once_with("m1")


# ------------------------------------------------------- files after upload


def test_hub_files_prints_one_line_per_file_with_its_state(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    listing = {
        "model_id": "m1",
        "files": [
            {
                "id": "f1",
                "file_name": "w.safetensors",
                "scan_status": "clean",
                "parameter_count": 7,
            },
            {"id": "f2", "file_name": "x.pt", "scan_status": "pending"},
            {
                "id": "f3",
                "file_name": "y.pt",
                "scan_status": "rejected",
                "rejection": {"code": "contains_code", "message": "it runs code"},
            },
            {"id": "f4", "file_name": "old.onnx"},
            "not a file",
        ],
    }
    fake = SimpleNamespace(list_files=mock.Mock(return_value=listing))
    with mock.patch("dagnam.hub", fake):
        assert run_cli(["hub", "files", "m1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "f1  w.safetensors  clean  7 parameters",
        "f2  x.pt  pending",
        "f3  y.pt  rejected  contains_code: it runs code",
        "f4  old.onnx  clean",
    ]
    fake.list_files.assert_called_once_with("m1")


def test_hub_files_with_no_files_says_so(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list_files=mock.Mock(return_value={"files": "nope"}))
    with mock.patch("dagnam.hub", fake):
        assert run_cli(["hub", "files", "m1"]) == 0
    assert capsys.readouterr().out.strip() == "No files."


def test_hub_files_json_prints_the_listing(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(list_files=mock.Mock(return_value={"files": [{"id": "f1"}]}))
    with mock.patch("dagnam.hub", fake):
        assert run_cli(["hub", "files", "m1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"files": [{"id": "f1"}]}


def test_hub_delete_file_succeeds_silently(run_cli: CliRunner, capsys: StrCapture) -> None:
    fake = SimpleNamespace(delete_file=mock.Mock(return_value=None))
    with mock.patch("dagnam.hub", fake):
        assert run_cli(["hub", "delete-file", "m1", "f1"]) == 0
    assert capsys.readouterr().out == ""
    fake.delete_file.assert_called_once_with("m1", "f1")


def test_hub_delete_file_not_found_says_file_or_model(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    from dagnam._core.exceptions import HubModelNotFoundError

    fake = SimpleNamespace(delete_file=mock.Mock(side_effect=HubModelNotFoundError("m1")))
    with mock.patch("dagnam.hub", fake), pytest.raises(SystemExit) as raised:
        run_cli(["hub", "delete-file", "m1", "f1"])
    assert raised.value.code == 1
    err = capsys.readouterr().err
    assert "File or model not found" in err
    assert "dagnam hub files m1" in err


def test_hub_link_design_sends_a_design_file(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    design = {"diagram_state": {"nodes": []}, "architecture_config": {"layers": []}}
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    fake = SimpleNamespace(link_design=mock.Mock(return_value={"id": "v1"}))
    with mock.patch("dagnam.hub", fake):
        assert (
            run_cli(
                ["hub", "link-design", "m1", "--version-id", "v1", "--architecture", f"@{path}"]
            )
            == 0
        )
    fake.link_design.assert_called_once_with(
        "m1", "v1", project_id=None, architecture_version_id=None, architecture=design
    )
    assert json.loads(capsys.readouterr().out) == {"id": "v1"}


def test_hub_link_design_sends_a_project(run_cli: CliRunner) -> None:
    fake = SimpleNamespace(link_design=mock.Mock(return_value={"id": "v1"}))
    argv = ["hub", "link-design", "m1", "--version-id", "v1", "--project", "p1"]
    with mock.patch("dagnam.hub", fake):
        assert run_cli([*argv, "--architecture-version", "a1"]) == 0
    fake.link_design.assert_called_once_with(
        "m1", "v1", project_id="p1", architecture_version_id="a1", architecture=None
    )


def test_hub_link_design_needs_exactly_one_source(run_cli: CliRunner, capsys: StrCapture) -> None:
    from dagnam.resources import hub_files

    fake = SimpleNamespace(link_design=hub_files.link_design)
    with mock.patch("dagnam.hub", fake), pytest.raises(SystemExit) as raised:
        run_cli(["hub", "link-design", "m1", "--version-id", "v1"])
    assert raised.value.code == 2
    assert "exactly one of project_id or architecture" in capsys.readouterr().err


def test_hub_link_design_needs_a_version(run_cli: CliRunner) -> None:
    with pytest.raises(SystemExit) as raised:
        run_cli(["hub", "link-design", "m1", "--project", "p1"])
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["hub", "publish", "--help"],
        ["hub", "version", "create", "--help"],
        ["hub", "upload-file", "--help"],
    ],
)
def test_the_file_help_lists_every_accepted_kind(
    run_cli: CliRunner, capsys: StrCapture, argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        run_cli(argv)
    assert raised.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert ".safetensors, .onnx, .pt, .pth, .h5" in out
    assert "config.json, tokenizer.json, tokenizer_config.json" in out
    assert "generation_config.json, special_tokens_map.json" in out
