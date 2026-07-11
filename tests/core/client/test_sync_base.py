"""Wire-level coverage for sync client base helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from tests.typing_helpers import PytestMonkeyPatch

from dagnam._core.client import DagnamClient, base as base_mod
from dagnam._core.client.base import (
    BaseDagnamClient,
    _progress_disabled,
    _sanitize_filename,
    content_disposition_safe_name,
    is_redirect_response,
    is_success_response,
    parse_content_disposition_filename,
    safe_error_body_from_response,
)
from dagnam._core.exceptions import APIError, AuthError

API = "https://api.test"


def _tty(monkeypatch: PytestMonkeyPatch, *, is_tty: bool) -> None:
    monkeypatch.setattr(base_mod.sys, "stderr", SimpleNamespace(isatty=lambda: is_tty))


# ---------------------------------------------------------------- progress bar gating


def test_progress_disabled_without_total(monkeypatch: PytestMonkeyPatch) -> None:
    _tty(monkeypatch, is_tty=True)
    assert _progress_disabled(None, show_progress=True) is True


def test_progress_disabled_when_opted_out(monkeypatch: PytestMonkeyPatch) -> None:
    _tty(monkeypatch, is_tty=True)
    assert _progress_disabled(10, show_progress=False) is True


def test_progress_disabled_in_non_tty(monkeypatch: PytestMonkeyPatch) -> None:
    # A carriage-return progress bar in CI logs / notebooks / a pipe is just spam.
    _tty(monkeypatch, is_tty=False)
    assert _progress_disabled(10, show_progress=True) is True


def test_progress_enabled_in_tty(monkeypatch: PytestMonkeyPatch) -> None:
    _tty(monkeypatch, is_tty=True)
    assert _progress_disabled(10, show_progress=True) is False


# ---------------------------------------------------------------- streaming download cleanup


def test_stream_response_to_file_closes_response_on_write_error(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    # If a chunk write fails mid-body (e.g. disk full), the streaming response
    # must still be closed so the underlying connection is released.
    _tty(monkeypatch, is_tty=False)
    resp = MagicMock()
    resp.headers = {"Content-Length": "10"}

    def chunks() -> object:
        yield b"partial"
        raise OSError("disk full")

    resp.iter_content = MagicMock(return_value=chunks())
    with pytest.raises(OSError, match="disk full"):
        BaseDagnamClient._stream_response_to_file(resp, tmp_path / "o.bin", show_progress=False)
    resp.close.assert_called_once()


def test_append_stream_to_file_closes_response_on_write_error(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    _tty(monkeypatch, is_tty=False)
    dest = tmp_path / "o.bin"
    dest.write_bytes(b"existing")
    resp = MagicMock()
    resp.headers = {"Content-Length": "10"}

    def chunks() -> object:
        yield b"more"
        raise OSError("disk full")

    resp.iter_content = MagicMock(return_value=chunks())
    with pytest.raises(OSError, match="disk full"):
        BaseDagnamClient._append_stream_to_file(resp, dest, show_progress=False)
    resp.close.assert_called_once()


class _ErrorResponse:
    """Minimal requests.Response stand-in for `_raise_for_status` error paths."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.ok = False
        self.headers: dict[str, str] = {"Content-Type": "text/plain"}
        self.content = b"boom"
        self.text = "boom"


# ---------------------------------------------------------------- base helpers


def testparse_content_disposition_filename_quoted() -> None:
    assert parse_content_disposition_filename('attachment; filename="cool.csv"') == "cool.csv"


def testparse_content_disposition_filename_unquoted() -> None:
    assert parse_content_disposition_filename("attachment; filename=cool.csv") == "cool.csv"


def testparse_content_disposition_filename_default_when_none() -> None:
    assert parse_content_disposition_filename(None) == "data"


def testparse_content_disposition_filename_default_when_no_filename_param() -> None:
    assert parse_content_disposition_filename("inline") == "data"


def test_sanitize_filename_rejects_path_separator() -> None:
    with pytest.raises(ValueError, match="Unsafe filename"):
        _sanitize_filename("../etc/passwd")


def test_sanitize_filename_rejects_windows_reserved() -> None:
    with pytest.raises(ValueError, match="Unsafe filename"):
        _sanitize_filename("CON.txt")


def test_sanitize_filename_rejects_drive_letter() -> None:
    with pytest.raises(ValueError, match="Unsafe filename"):
        _sanitize_filename("C:nasty")


def test_sanitize_filename_rejects_empty_and_dots() -> None:
    for bad in ("", ".", ".."):
        with pytest.raises(ValueError):
            _sanitize_filename(bad)


# ------------------------------------------------------ content_disposition_safe_name


def test_content_disposition_safe_name_quoted() -> None:
    header = 'attachment; filename="dagnam_export_u1.zip"'
    assert content_disposition_safe_name(header, default="export.zip") == "dagnam_export_u1.zip"


def test_content_disposition_safe_name_unquoted() -> None:
    header = "attachment; filename=dagnam_export_u1.zip"
    assert content_disposition_safe_name(header, default="export.zip") == "dagnam_export_u1.zip"


def test_content_disposition_safe_name_absent_header_uses_default() -> None:
    assert content_disposition_safe_name(None, default="export.zip") == "export.zip"


def test_content_disposition_safe_name_no_filename_param_uses_default() -> None:
    assert content_disposition_safe_name("inline", default="export.zip") == "export.zip"


def test_content_disposition_safe_name_strips_traversal_to_basename() -> None:
    header = 'attachment; filename="../../etc/passwd"'
    assert content_disposition_safe_name(header, default="export.zip") == "passwd"


def test_content_disposition_safe_name_strips_windows_separators() -> None:
    header = r'attachment; filename="..\..\windows\system32\evil.dll"'
    assert content_disposition_safe_name(header, default="export.zip") == "evil.dll"


def test_content_disposition_safe_name_empty_quoted_uses_default() -> None:
    header = 'attachment; filename=""'
    assert content_disposition_safe_name(header, default="export.zip") == "export.zip"


def test_content_disposition_safe_name_dot_uses_default() -> None:
    header = 'attachment; filename="."'
    assert content_disposition_safe_name(header, default="export.zip") == "export.zip"


def test_content_disposition_safe_name_dotdot_uses_default() -> None:
    header = 'attachment; filename=".."'
    assert content_disposition_safe_name(header, default="export.zip") == "export.zip"


def test_content_disposition_safe_name_strips_drive_letter() -> None:
    # Critical regression: a bare drive-letter prefix has no "/" or "\\" for
    # PurePosixPath(...).name to strip, so it must be stripped separately.
    header = 'attachment; filename="D:evil.dll"'
    result = content_disposition_safe_name(header, default="export.zip")
    assert result == "evil.dll"
    assert not any(sep in result for sep in (":", "/", "\\"))


def test_content_disposition_safe_name_strips_multi_colon() -> None:
    header = 'attachment; filename="a:b:c.txt"'
    result = content_disposition_safe_name(header, default="export.zip")
    assert result == "c.txt"
    assert not any(sep in result for sep in (":", "/", "\\"))


def test_content_disposition_safe_name_strips_non_drive_colon_prefix() -> None:
    header = 'attachment; filename="my:file.txt"'
    result = content_disposition_safe_name(header, default="export.zip")
    assert result == "file.txt"
    assert not any(sep in result for sep in (":", "/", "\\"))


def test_content_disposition_safe_name_strips_drive_and_backslashes() -> None:
    header = 'attachment; filename="C:\\Windows\\system32\\evil.dll"'
    result = content_disposition_safe_name(header, default="export.zip")
    assert result == "evil.dll"
    assert not any(sep in result for sep in (":", "/", "\\"))


def test_content_disposition_safe_name_reserved_device_stem_uses_default() -> None:
    header = 'attachment; filename="con.txt"'
    assert content_disposition_safe_name(header, default="export.zip") == "export.zip"


def test_content_disposition_safe_name_bare_reserved_device_name_uses_default() -> None:
    header = 'attachment; filename="NUL"'
    assert content_disposition_safe_name(header, default="export.zip") == "export.zip"


def test_content_disposition_safe_name_bare_drive_letter_uses_default() -> None:
    header = 'attachment; filename="D:"'
    assert content_disposition_safe_name(header, default="export.zip") == "export.zip"


def testis_success_response_from_status_code() -> None:
    class _R:
        status_code = 204
        ok = False

    assert is_success_response(_R())


def testis_success_response_falls_back_to_ok_attr() -> None:
    class _R:
        status_code = None
        ok = True

    assert is_success_response(_R())


class _RedirectStub:
    """Minimal ResponseLike for is_redirect_response branch coverage."""

    text = ""
    content = b""

    def __init__(self, status_code: object, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers


def test_is_redirect_response_3xx_with_location() -> None:
    stub = _RedirectStub(307, {"Location": "https://s3/presigned"})
    assert is_redirect_response(stub)


def test_is_redirect_response_3xx_without_location() -> None:
    assert not is_redirect_response(_RedirectStub(307, {}))


def test_is_redirect_response_non_3xx() -> None:
    assert not is_redirect_response(_RedirectStub(200, {"Location": "https://s3/x"}))


def test_is_redirect_response_non_int_status() -> None:
    assert not is_redirect_response(_RedirectStub(None, {"Location": "https://s3/x"}))


def test_safe_error_body_from_response_delegates_to_common(client: DagnamClient) -> None:
    class _R:
        headers: ClassVar[dict[str, str]] = {"Content-Type": "text/plain"}
        content = b"err"
        text = "err"

    # Intentionally partial response fake (no status_code); the helper only
    # touches headers/text/content.
    assert safe_error_body_from_response(_R()) == "err"  # pyright: ignore[reportArgumentType]


def test_raise_for_status_maps_401_to_autherror() -> None:
    with pytest.raises(AuthError, match="invalid or expired API key"):
        # Partial response fake exercising only the status-code mapping.
        BaseDagnamClient._raise_for_status(_ErrorResponse(401), "ds-1")  # pyright: ignore[reportArgumentType]


def test_raise_for_status_maps_other_codes_to_apierror() -> None:
    with pytest.raises(APIError) as exc_info:
        BaseDagnamClient._raise_for_status(_ErrorResponse(500), "ds-1")  # pyright: ignore[reportArgumentType]
    assert exc_info.value.status_code == 500
