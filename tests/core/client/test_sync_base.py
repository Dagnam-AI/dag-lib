"""Wire-level coverage for sync client base helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import (
    BaseDagnamClient,
    _sanitize_filename,
    is_success_response,
    parse_content_disposition_filename,
    safe_error_body_from_response,
)
from dagnam._core.exceptions import APIError, AuthError

API = "https://api.test"


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
