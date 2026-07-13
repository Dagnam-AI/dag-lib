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
    safe_download_basename,
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


# ---------------------------------------------------------------- download size cap


class _FakeResp:
    """Minimal streaming-response stand-in for the base writers."""

    def __init__(self, chunks: list[bytes], content_length: int | None) -> None:
        self._chunks = chunks
        self.headers: dict[str, str] = (
            {} if content_length is None else {"Content-Length": str(content_length)}
        )

    def iter_content(self, chunk_size: int) -> object:
        yield from self._chunks

    def close(self) -> None:
        pass


def testresolve_max_download_bytes_default(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(base_mod, "get_config_value", lambda _k, default: default)
    assert base_mod.resolve_max_download_bytes() == base_mod.DEFAULT_MAX_DOWNLOAD_BYTES


@pytest.mark.parametrize("bad", [True, "10", 0, -5, 1.5])
def testresolve_max_download_bytes_rejects_non_positive_int(
    monkeypatch: PytestMonkeyPatch, bad: object
) -> None:
    monkeypatch.setattr(base_mod, "get_config_value", lambda _k, _default: bad)
    assert base_mod.resolve_max_download_bytes() == base_mod.DEFAULT_MAX_DOWNLOAD_BYTES


def testresolve_max_download_bytes_honours_valid_int(monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(base_mod, "get_config_value", lambda _k, _default: 4096)
    assert base_mod.resolve_max_download_bytes() == 4096


def test_stream_rejects_oversized_content_length(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(base_mod, "resolve_max_download_bytes", lambda: 4)
    dest = tmp_path / "big.bin"
    with pytest.raises(base_mod.DownloadTooLargeError):
        BaseDagnamClient._stream_response_to_file(
            _FakeResp([b"xxxxxxxx"], content_length=8),  # pyright: ignore[reportArgumentType]
            dest,
            show_progress=False,
        )
    assert not dest.exists()


def test_stream_aborts_when_body_exceeds_cap(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(base_mod, "resolve_max_download_bytes", lambda: 4)
    dest = tmp_path / "big.bin"
    with pytest.raises(base_mod.DownloadTooLargeError):
        BaseDagnamClient._stream_response_to_file(
            _FakeResp([b"xx", b"xx", b"xx"], content_length=None),  # pyright: ignore[reportArgumentType]
            dest,
            show_progress=False,
        )
    assert not dest.exists()


def test_stream_writes_within_cap(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(base_mod, "resolve_max_download_bytes", lambda: 1024)
    dest = tmp_path / "ok.bin"
    BaseDagnamClient._stream_response_to_file(
        _FakeResp([b"ab", b"cd"], content_length=4),  # pyright: ignore[reportArgumentType]
        dest,
        show_progress=False,
    )
    assert dest.read_bytes() == b"abcd"


def test_append_rejects_oversized_resumed_total(
    tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(base_mod, "resolve_max_download_bytes", lambda: 6)
    dest = tmp_path / "part.bin"
    dest.write_bytes(b"aaaa")  # 4 already written; +4 incoming would exceed 6
    with pytest.raises(base_mod.DownloadTooLargeError):
        BaseDagnamClient._append_stream_to_file(
            _FakeResp([b"bbbb"], content_length=4),  # pyright: ignore[reportArgumentType]
            dest,
            show_progress=False,
        )
    assert not dest.exists()


def test_append_aborts_mid_stream_over_cap(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(base_mod, "resolve_max_download_bytes", lambda: 5)
    dest = tmp_path / "part.bin"
    dest.write_bytes(b"aa")  # 2 written; streaming 4 more crosses 5 with no Content-Length
    with pytest.raises(base_mod.DownloadTooLargeError):
        BaseDagnamClient._append_stream_to_file(
            _FakeResp([b"bb", b"bb"], content_length=None),  # pyright: ignore[reportArgumentType]
            dest,
            show_progress=False,
        )
    assert not dest.exists()


def test_append_within_cap_succeeds(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    monkeypatch.setattr(base_mod, "resolve_max_download_bytes", lambda: 1024)
    dest = tmp_path / "part.bin"
    dest.write_bytes(b"aa")
    BaseDagnamClient._append_stream_to_file(
        _FakeResp([b"bb"], content_length=2),  # pyright: ignore[reportArgumentType]
        dest,
        show_progress=False,
    )
    assert dest.read_bytes() == b"aabb"


# ---------------------------------------------------------------- token scrub


@pytest.mark.parametrize(
    ("raw", "expected_absent", "expected_present"),
    [
        ("url: /x?token=SECRET", "SECRET", "token=***"),
        ("GET /y?X-Amz-Signature=ABC&z=1 failed", "ABC", "Signature=***"),
        ("/z?credential=CRED", "CRED", "credential=***"),
        ("plain message no query", "", "plain message no query"),
    ],
)
def testscrub_secret_params(raw: str, expected_absent: str, expected_present: str) -> None:
    scrubbed = base_mod.scrub_secret_params(raw)
    if expected_absent:
        assert expected_absent not in scrubbed
    assert expected_present in scrubbed


def test_get_stream_scrubs_token_on_connection_error(monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise base_mod.requests.ConnectionError(
            "HTTPSConnectionPool(host='api.test', port=443): Max retries exceeded "
            "with url: /x?token=SECRET (Caused by NewConnectionError)"
        )

    monkeypatch.setattr(base_mod.requests, "get", _boom)
    client = DagnamClient(API, "k")
    with pytest.raises(APIError) as ei:
        client._get_stream(f"{API}/x?token=SECRET")
    assert "SECRET" not in str(ei.value)
    assert "token=***" in str(ei.value)


def test_get_stream_no_auth_scrubs_token_on_timeout(monkeypatch: PytestMonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise base_mod.requests.Timeout("timed out for url: /obj?X-Amz-Signature=SECRET")

    monkeypatch.setattr(base_mod.requests, "get", _boom)
    with pytest.raises(APIError) as ei:
        BaseDagnamClient._get_stream_no_auth(f"{API}/obj?X-Amz-Signature=SECRET")
    assert "SECRET" not in str(ei.value)


# ---------------------------------------------------------------- safe_download_basename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("data.csv", "data.csv"),  # legit basename preserved
        ("/home/u/.bashrc", ".bashrc"),  # absolute -> basename
        ("../../../etc/passwd", "passwd"),  # traversal -> basename
        ("a/b/c/file.bin", "file.bin"),  # nested -> basename
        ("C:\\Windows\\evil.exe", "evil.exe"),  # windows drive + backslash
        ("name:stream", "stream"),  # NTFS ADS prefix stripped
        ("..", "DEF"),  # reduces to nothing usable -> default
        ("", "DEF"),  # empty -> default
        ("nul", "DEF"),  # windows reserved device -> default
        ("con.txt", "DEF"),  # reserved stem -> default
    ],
)
def test_safe_download_basename(raw: str, expected: str) -> None:
    assert safe_download_basename(raw, default="DEF") == expected
