"""Wire-level coverage for sync client base helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.client.base import (
    _sanitize_filename,
    is_success_response,
    parse_content_disposition_filename,
    safe_error_body_from_response,
)

API = "https://api.test"


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
