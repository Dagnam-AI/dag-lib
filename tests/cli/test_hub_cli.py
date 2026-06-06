"""CLI hub subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, StrCapture


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
