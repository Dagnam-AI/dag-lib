"""Tests for presigned URL downloads and resumable downloads."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dagnam._core.aio.base import _parse_cd as _parse_async_filename
from dagnam._core.client import DagnamClient
from dagnam._core.client.base import _parse_filename


def _mock_response(status_code, headers, chunks):
    """Create a mock requests.Response for streaming."""
    resp = MagicMock()
    resp.ok = 200 <= status_code < 300
    resp.status_code = status_code
    resp.headers = headers
    resp.iter_content = MagicMock(return_value=iter(chunks))
    return resp


class TestSignedDownload:
    """Tests for presigned URL download (no auth header)."""

    def test_signed_download_omits_auth_header(self, tmp_path: Path):
        """When download_url is provided, auth header is omitted."""
        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
            },
            [b"hello"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            path = client.download_dataset("ds1", tmp_path, download_url="https://signed.test/file")

        assert path.read_bytes() == b"hello"
        # Verify no auth header was sent
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs.get("headers") == {}

    def test_signed_download_uses_provided_url(self, tmp_path: Path):
        """Uses the provided download_url instead of constructing one."""
        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
            },
            [b"hello"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            client.download_dataset(
                "ds1", tmp_path, download_url="https://signed.test/file?token=abc"
            )

        # Verify the signed URL was used
        call_args = mock_get.call_args
        assert call_args.args[0] == "https://signed.test/file?token=abc"

    def test_normal_download_includes_auth_header(self, tmp_path: Path):
        """Normal download (no download_url) includes auth header."""
        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
            },
            [b"hello"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            client.download_dataset("ds1", tmp_path)

        call_kwargs = mock_get.call_args.kwargs
        assert "Authorization" in call_kwargs.get("headers", {})


class TestResumableDownload:
    """Tests for resumable download with Range header."""

    def test_full_download_writes_part_then_renames(self, tmp_path: Path):
        """A full download writes to .part first, then atomically renames."""
        client = DagnamClient("https://api.test", "secret")

        def chunks():
            assert (tmp_path / "data.csv.part").exists()
            assert not (tmp_path / "data.csv").exists()
            yield b"hello"

        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
            },
            chunks(),
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp):
            path = client.download_dataset("ds1", tmp_path, filename="data.csv")

        assert path == tmp_path / "data.csv"
        assert path.read_bytes() == b"hello"
        assert not (tmp_path / "data.csv.part").exists()

    def test_resume_sends_range_and_appends(self, tmp_path: Path):
        """Resumes download by sending Range header and appending to .part file."""
        # Create a partial download
        part_file = tmp_path / "data.csv.part"
        part_file.write_bytes(b"hello ")

        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            206,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
                "Content-Range": "bytes 6-10/11",
            },
            [b"world"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            path = client.download_dataset("ds1", tmp_path, filename="data.csv", resume=True)

        # Verify Range header was sent
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["headers"]["Range"] == "bytes=6-"

        # Verify content was appended
        assert path.read_bytes() == b"hello world"

    def test_resume_restarts_on_200(self, tmp_path: Path):
        """If server returns 200 instead of 206, restart full download."""
        # Create a partial download
        part_file = tmp_path / "data.csv.part"
        part_file.write_bytes(b"old partial data")

        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "11",
            },
            [b"hello world"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp):
            path = client.download_dataset("ds1", tmp_path, filename="data.csv", resume=True)

        # Full content replaces partial
        assert path.read_bytes() == b"hello world"

    def test_no_resume_when_no_part_file(self, tmp_path: Path):
        """No Range header when there's no .part file."""
        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
            },
            [b"hello"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp) as mock_get:
            path = client.download_dataset("ds1", tmp_path, filename="data.csv", resume=True)

        # No Range header
        call_kwargs = mock_get.call_args.kwargs
        assert "Range" not in call_kwargs.get("headers", {})
        assert path.read_bytes() == b"hello"

    def test_resume_disabled_ignores_part_file(self, tmp_path: Path):
        """When resume=False, ignores existing .part file."""
        part_file = tmp_path / "data.csv.part"
        part_file.write_bytes(b"old data")

        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Length": "5",
            },
            [b"fresh"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp):
            path = client.download_dataset("ds1", tmp_path, filename="data.csv", resume=False)

        assert path.read_bytes() == b"fresh"
        # Part file should be cleaned up
        assert not part_file.exists()

    def test_filename_from_content_disposition(self, tmp_path: Path):
        """Extracts filename from Content-Disposition header."""
        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="my_data.csv"',
                "Content-Length": "5",
            },
            [b"hello"],
        )
        with patch("dagnam._core.client.base.requests.get", return_value=resp):
            path = client.download_dataset("ds1", tmp_path)

        assert path.name == "my_data.csv"

    def test_content_disposition_path_traversal_is_rejected(self, tmp_path: Path):
        """Server-provided filenames cannot escape the output directory."""
        client = DagnamClient("https://api.test", "secret")
        resp = _mock_response(
            200,
            {
                "Content-Disposition": 'attachment; filename="../../../escape.txt"',
                "Content-Length": "5",
            },
            [b"hello"],
        )

        with patch("dagnam._core.client.base.requests.get", return_value=resp):
            with pytest.raises(ValueError, match="Unsafe filename"):
                client.download_dataset("ds1", tmp_path)

        assert not (tmp_path.parent / "escape.txt").exists()


class TestContentDispositionFilename:
    def test_rejects_empty_dot_and_parent_names(self):
        for value in ('attachment; filename=""', "attachment; filename=.", "filename=.."):
            with pytest.raises(ValueError, match="Unsafe filename"):
                _parse_filename(value)

    def test_rejects_slash_and_backslash_paths(self):
        for value in ('attachment; filename="../x.csv"', 'attachment; filename="..\\x.csv"'):
            with pytest.raises(ValueError, match="Unsafe filename"):
                _parse_filename(value)

    def test_rejects_windows_special_paths(self):
        for value in (
            'attachment; filename="C:escape.txt"',
            'attachment; filename="file.txt:ads"',
            'attachment; filename="CON"',
            'attachment; filename="nul.txt"',
        ):
            with pytest.raises(ValueError, match="Unsafe filename"):
                _parse_filename(value)

    def test_async_parser_rejects_windows_special_paths(self):
        for value in (
            'attachment; filename="C:escape.txt"',
            'attachment; filename="file.txt:ads"',
            'attachment; filename="CON"',
            'attachment; filename="nul.txt"',
        ):
            with pytest.raises(ValueError, match="Unsafe filename"):
                _parse_async_filename(value)
