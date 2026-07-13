"""Transport mapping + connect-phase retry for the system-dataset artifact download."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

from dagnam._core._retry import RetryBudget
from dagnam._core.exceptions import APIError
from dagnam.data.loaders.system import dispatch

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker


def test_download_artifact_retries_transient_connect(
    tmp_path: Path, requests_mock: RequestsMocker, monkeypatch: PytestMonkeyPatch
) -> None:
    url = "https://cdn.test/artifact.bin"
    requests_mock.get(url, [{"status_code": 503}, {"status_code": 200, "content": b"DATA"}])
    monkeypatch.setattr(dispatch, "_RETRY_SLEEP", lambda _s: None)
    monkeypatch.setattr(dispatch, "_RETRY_RNG", lambda: 1.0)
    monkeypatch.setattr(dispatch, "_DISPATCH_BUDGET", RetryBudget())
    dest = tmp_path / "a.bin"
    dispatch._download_artifact(url, dest)  # type: ignore[attr-defined]
    assert dest.read_bytes() == b"DATA"
    assert requests_mock.call_count == 2


def test_download_artifact_maps_http_error(tmp_path: Path, requests_mock: RequestsMocker) -> None:
    url = "https://cdn.test/artifact.bin"
    requests_mock.get(url, status_code=404)
    with pytest.raises(APIError):
        dispatch._download_artifact(url, tmp_path / "a.bin")  # type: ignore[attr-defined]
    assert requests_mock.call_count == 1  # 404 is non-transient → not retried


def test_download_artifact_maps_transport_error_and_cleans_tmp(
    tmp_path: Path, requests_mock: RequestsMocker, monkeypatch: PytestMonkeyPatch
) -> None:
    url = "https://cdn.test/artifact.bin"
    requests_mock.get(url, exc=requests.ConnectionError("down"))
    monkeypatch.setattr(dispatch, "_RETRY_SLEEP", lambda _s: None)
    monkeypatch.setattr(dispatch, "_DISPATCH_BUDGET", RetryBudget())
    dest = tmp_path / "a.bin"
    with pytest.raises(APIError, match="Artifact download failed"):
        dispatch._download_artifact(url, dest)  # type: ignore[attr-defined]
    assert not (tmp_path / "a.bin.tmp").exists()  # tmp cleaned up on failure
