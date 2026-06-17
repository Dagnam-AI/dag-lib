"""Unit tests for dagnam.codegen module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pytest

from dagnam import codegen
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import CodegenError, CodegenValidationError
from dagnam._core.lro import LongRunningOperation


def _client(**overrides: object) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.configure_mock(**overrides)
    return client


class TestGenerate:
    def test_sync_returns_dict(self) -> None:
        c = _client(generate_code=MagicMock(return_value={"task_id": "t1", "code": "..."}))
        out = codegen.generate("p1", client=c)
        c.generate_code.assert_called_once_with(
            "p1", framework="pytorch", version_id=None, async_mode=False
        )
        assert isinstance(out, dict)
        assert out["task_id"] == "t1"

    def test_async_returns_lro(self) -> None:
        c = _client(generate_code=MagicMock(return_value={"task_id": "t1", "status": "pending"}))
        op = codegen.generate("p1", async_mode=True, client=c)
        c.generate_code.assert_called_once_with(
            "p1", framework="pytorch", version_id=None, async_mode=True
        )
        assert isinstance(op, LongRunningOperation)
        initial = op.initial()
        assert initial is not None
        assert initial["task_id"] == "t1"

    def test_passes_framework_and_version(self) -> None:
        c = _client(generate_code=MagicMock(return_value={"task_id": "t1"}))
        codegen.generate("p1", framework="tensorflow", version_id="v2", client=c)
        c.generate_code.assert_called_once_with(
            "p1", framework="tensorflow", version_id="v2", async_mode=False
        )

    def test_async_missing_string_task_id_raises(self) -> None:
        # async_mode response without a string task_id is rejected before the LRO.
        c = _client(generate_code=MagicMock(return_value={"status": "pending"}))
        with pytest.raises(ValueError, match="did not include a string task_id"):
            codegen.generate("p1", async_mode=True, client=c)


class TestPreview:
    def test_preview_delegates(self) -> None:
        c = _client(preview_code=MagicMock(return_value={"code": "import torch"}))
        out = codegen.preview("p1", framework="pytorch", client=c)
        c.preview_code.assert_called_once_with("p1", framework="pytorch", version_id=None)
        assert isinstance(out, dict)
        assert out["code"] == "import torch"


class TestValidate:
    def test_validate_delegates(self) -> None:
        c = _client(validate_code=MagicMock(return_value={"valid": True}))
        out = codegen.validate("p1", version_id="v1", client=c)
        c.validate_code.assert_called_once_with("p1", version_id="v1")
        assert out["valid"] is True


class TestDownload:
    def test_download_returns_bytes_when_no_dest(self) -> None:
        c = _client(download_code=MagicMock(return_value=b"code"))
        out = codegen.download("p1", client=c)
        c.download_code.assert_called_once_with(
            "p1", framework="pytorch", version_id=None, dest_path=None, show_progress=True
        )
        assert out == b"code"

    def test_download_writes_to_dest(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.zip"
        c = _client(download_code=MagicMock(return_value=dest))
        result = codegen.download("p1", dest=dest, client=c)
        c.download_code.assert_called_once_with(
            "p1", framework="pytorch", version_id=None, dest_path=dest, show_progress=True
        )
        assert result == dest

    def test_download_with_dest_rejects_non_path(self, tmp_path: Path) -> None:
        # dest given but the client returned bytes instead of a path → TypeError.
        dest = tmp_path / "out.zip"
        c = _client(download_code=MagicMock(return_value=b"code"))
        with pytest.raises(TypeError, match="Expected generated code download path"):
            codegen.download("p1", dest=dest, client=c)

    def test_download_without_dest_rejects_non_bytes(self, tmp_path: Path) -> None:
        # no dest but the client returned a path instead of bytes → TypeError.
        c = _client(download_code=MagicMock(return_value=tmp_path / "x.zip"))
        with pytest.raises(TypeError, match="Expected generated code bytes"):
            codegen.download("p1", client=c)

    def test_download_to_existing_directory_extracts_generated_code(self, tmp_path: Path) -> None:
        # dest is an existing directory → download zip to temp file and extract into dest.
        dest = tmp_path / "pytorch"
        dest.mkdir()

        def _fake_download(*_args: object, dest_path: Path, **_kwargs: object) -> Path:
            with zipfile.ZipFile(dest_path, "w") as zf:
                zf.writestr("model.py", "import torch\n")
            return dest_path

        c = _client(download_code=MagicMock(side_effect=_fake_download))
        result = codegen.download("p1", framework="pytorch", dest=dest, client=c)

        assert result == dest
        assert (dest / "model.py").read_text() == "import torch\n"
        # The temp archive path passed to download_code is not the dest directory.
        call = c.download_code.call_args
        assert call.kwargs["dest_path"] != dest
        assert call.kwargs["dest_path"].suffix == ".zip"

    def test_download_to_directory_rejects_non_path_archive(self, tmp_path: Path) -> None:
        # dest is a directory but download_code returns bytes instead of a temp path → TypeError.
        dest = tmp_path / "pytorch"
        dest.mkdir()
        c = _client(download_code=MagicMock(return_value=b"not-a-path"))
        with pytest.raises(TypeError, match="Expected a download path for the temporary archive"):
            codegen.download("p1", dest=dest, client=c)


class TestStatus:
    def test_status_delegates(self) -> None:
        c = _client(get_code_status=MagicMock(return_value={"status": "completed"}))
        out = codegen.status("p1", "t1", client=c)
        c.get_code_status.assert_called_once_with("p1", "t1")
        assert out["status"] == "completed"


class TestErrorPropagation:
    def test_generate_propagates_codegenerror(self) -> None:
        c = _client()
        c.generate_code.side_effect = CodegenError("boom")
        with pytest.raises(CodegenError):
            codegen.generate("p1", client=c)

    def test_validate_propagates_validationerror(self) -> None:
        c = _client(validate_code=MagicMock(side_effect=CodegenValidationError("invalid arch")))
        with pytest.raises(CodegenValidationError):
            codegen.validate("p1", client=c)
