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
    fake.upload_file.assert_called_once_with("m1", str(f))


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
                    "--file",
                    str(weights),
                ]
            )
            == 1
        )
    assert "halted" in capsys.readouterr().err


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
