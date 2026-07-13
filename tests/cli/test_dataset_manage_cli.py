"""CLI coverage for `dagnam dataset preview / update / delete / roles`."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from dagnam.cli.dataset import _decode_image_bytes

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture

_PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"
_WEBP = b"RIFF\x00\x00\x00\x00WEBPfake"


# ------------------------------------------------------- _decode_image_bytes


def test_decode_image_bytes_data_uri() -> None:
    uri = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    decoded = _decode_image_bytes(uri)
    assert decoded is not None
    assert decoded == (_PNG, "png")


def test_decode_image_bytes_webp() -> None:
    decoded = _decode_image_bytes(base64.b64encode(_WEBP).decode())
    assert decoded is not None
    assert decoded[1] == "webp"


def test_decode_image_bytes_rejects_non_image_base64() -> None:
    assert _decode_image_bytes(base64.b64encode(b"just plain text bytes").decode()) is None


def test_decode_image_bytes_rejects_invalid_base64() -> None:
    assert _decode_image_bytes("not-valid-base64!!!") is None


def test_decode_image_bytes_rejects_non_string() -> None:
    assert _decode_image_bytes(12345) is None


def test_decode_image_bytes_rejects_oversized_payload(monkeypatch: PytestMonkeyPatch) -> None:
    # A hostile server returning a multi-GB base64 blob must not be decoded into
    # memory; the size cap rejects it before base64.b64decode runs.
    from dagnam.cli import dataset as dataset_cli

    monkeypatch.setattr(dataset_cli, "_MAX_PREVIEW_IMAGE_B64_CHARS", 4)
    oversized = base64.b64encode(_PNG).decode()
    assert _decode_image_bytes(oversized) is None


# ------------------------------------------------------------------- preview


def test_preview_tabular_renders_table_and_statistics(
    run_cli: CliRunner, capsys: StrCapture
) -> None:
    payload = {"samples": [{"a": 1, "b": 2}], "statistics": {"count": 1}}
    with mock.patch("dagnam.preview_dataset", mock.Mock(return_value=payload)) as preview:
        run_cli(["dataset", "preview", "ds-1", "--rows", "5"])
    preview.assert_called_once_with("ds-1", rows=5)
    out = capsys.readouterr().out
    assert "a" in out
    assert "b" in out
    assert "Statistics:" in out
    assert "count: 1" in out


def test_preview_clamps_rows_above_range(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"samples": [], "statistics": {}}
    with mock.patch("dagnam.preview_dataset", mock.Mock(return_value=payload)) as preview:
        run_cli(["dataset", "preview", "ds-1", "--rows", "100000"])
    preview.assert_called_once_with("ds-1", rows=100)


def test_preview_decodes_image_samples(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path
) -> None:
    encoded = base64.b64encode(_PNG).decode()
    payload = {"samples": [{"image": encoded, "label": 3}], "statistics": {}}
    with mock.patch("dagnam.preview_dataset", mock.Mock(return_value=payload)):
        run_cli(["dataset", "preview", "ds-1", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Saved image to" in out
    saved = list(tmp_path.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == _PNG


def test_preview_multiple_rows_share_columns(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"samples": [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "statistics": {}}
    with mock.patch("dagnam.preview_dataset", mock.Mock(return_value=payload)):
        run_cli(["dataset", "preview", "ds-1"])
    out = capsys.readouterr().out
    assert "3" in out
    assert "4" in out


def test_preview_json_mode(run_cli: CliRunner, capsys: StrCapture) -> None:
    payload = {"samples": [{"a": 1}], "statistics": {"count": 1}}
    with mock.patch("dagnam.preview_dataset", mock.Mock(return_value=payload)):
        run_cli(["dataset", "preview", "ds-1", "--json"])
    assert json.loads(capsys.readouterr().out) == payload


def test_preview_empty_samples(run_cli: CliRunner, capsys: StrCapture) -> None:
    with mock.patch(
        "dagnam.preview_dataset", mock.Mock(return_value={"samples": [], "statistics": {}})
    ):
        run_cli(["dataset", "preview", "ds-1"])
    assert "No samples returned." in capsys.readouterr().out


# -------------------------------------------------------------------- update


def test_update_forwards_fields_and_prints_json(run_cli: CliRunner, capsys: StrCapture) -> None:
    result = {"id": "ds-1", "name": "new"}
    with mock.patch("dagnam.update_dataset", mock.Mock(return_value=result)) as update:
        run_cli(["dataset", "update", "ds-1", "--name", "new"])
    update.assert_called_once_with("ds-1", name="new", description=None, visibility=None)
    assert json.loads(capsys.readouterr().out) == result


# -------------------------------------------------------------------- delete


def test_delete_confirmed(run_cli: CliRunner, capsys: StrCapture) -> None:
    delete = mock.Mock()
    with (
        mock.patch("dagnam.delete_dataset", delete),
        mock.patch("builtins.input", return_value="yes"),
    ):
        run_cli(["dataset", "delete", "ds-1"])
    delete.assert_called_once_with("ds-1")
    assert "Dataset ds-1 deleted." in capsys.readouterr().out


def test_delete_yes_flag_bypasses_prompt(run_cli: CliRunner, capsys: StrCapture) -> None:
    delete = mock.Mock()

    def _boom(_prompt: str = "") -> str:
        raise AssertionError("input() must not be called when --yes is set")

    with (
        mock.patch("dagnam.delete_dataset", delete),
        mock.patch("builtins.input", _boom),
    ):
        run_cli(["dataset", "delete", "ds-1", "--yes"])
    delete.assert_called_once_with("ds-1")
    assert "deleted." in capsys.readouterr().out


def test_delete_aborted_on_typed_no(run_cli: CliRunner, capsys: StrCapture) -> None:
    delete = mock.Mock()
    with (
        mock.patch("dagnam.delete_dataset", delete),
        mock.patch("builtins.input", return_value="no"),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["dataset", "delete", "ds-1"])
    assert exc_info.value.code == 1
    assert "confirmation not received" in capsys.readouterr().err
    delete.assert_not_called()


# --------------------------------------------------------------------- roles


def test_roles_builds_dict_and_forwards(run_cli: CliRunner, capsys: StrCapture) -> None:
    result = {"column_roles": {"a": "target", "b": "feature"}, "roles_confirmed": True}
    with mock.patch("dagnam.update_dataset_roles", mock.Mock(return_value=result)) as roles:
        run_cli(
            [
                "dataset",
                "roles",
                "ds-1",
                "--set",
                "a=target",
                "--set",
                "b=feature",
                "--task-type-hint",
                "classification",
            ]
        )
    roles.assert_called_once_with(
        "ds-1", {"a": "target", "b": "feature"}, task_type_hint="classification"
    )
    assert json.loads(capsys.readouterr().out) == result


def test_roles_rejects_malformed_set(run_cli: CliRunner, capsys: StrCapture) -> None:
    roles = mock.Mock()
    with (
        mock.patch("dagnam.update_dataset_roles", roles),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_cli(["dataset", "roles", "ds-1", "--set", "bad"])
    assert exc_info.value.code == 1
    assert "Invalid --set" in capsys.readouterr().err
    roles.assert_not_called()
