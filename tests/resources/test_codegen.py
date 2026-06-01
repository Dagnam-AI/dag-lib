"""Unit tests for dagnam.codegen module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from dagnam import codegen
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import CodegenError, CodegenValidationError
from dagnam._core.lro import LongRunningOperation

if TYPE_CHECKING:
    from tests.typing_helpers import JsonObject


def _client(**overrides: JsonObject) -> MagicMock:
    return MagicMock(spec=DagnamClient, **overrides)


class TestGenerate:
    def test_sync_returns_dict(self) -> None:
        c = _client(generate_code=MagicMock(return_value={"task_id": "t1", "code": "..."}))
        out = codegen.generate("p1", client=c)
        c.generate_code.assert_called_once_with(
            "p1", framework="pytorch", version_id=None, async_mode=False
        )
        assert out["task_id"] == "t1"

    def test_async_returns_lro(self) -> None:
        c = _client(generate_code=MagicMock(return_value={"task_id": "t1", "status": "pending"}))
        op = codegen.generate("p1", async_mode=True, client=c)
        c.generate_code.assert_called_once_with(
            "p1", framework="pytorch", version_id=None, async_mode=True
        )
        assert isinstance(op, LongRunningOperation)
        assert op.initial()["task_id"] == "t1"

    def test_passes_framework_and_version(self) -> None:
        c = _client(generate_code=MagicMock(return_value={"task_id": "t1"}))
        codegen.generate("p1", framework="tensorflow", version_id="v2", client=c)
        c.generate_code.assert_called_once_with(
            "p1", framework="tensorflow", version_id="v2", async_mode=False
        )


class TestPreview:
    def test_preview_delegates(self) -> None:
        c = _client(preview_code=MagicMock(return_value={"code": "import torch"}))
        out = codegen.preview("p1", framework="pytorch", client=c)
        c.preview_code.assert_called_once_with("p1", framework="pytorch", version_id=None)
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
