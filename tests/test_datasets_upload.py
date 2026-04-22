"""Unit tests for dagnam.datasets_upload module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import datasets_upload
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import UploadError
from dagnam._core.lro import LongRunningOperation


def _client(**overrides) -> MagicMock:
    return MagicMock(spec=DagnamClient, **overrides)


class TestUpload:
    def test_upload_delegates(self):
        c = _client(upload_dataset=MagicMock(return_value={"id": "ds1"}))
        out = datasets_upload.upload(
            "data.csv", name="my-ds", dataset_type="tabular", format="csv",
            description="test", visibility="public", license="mit", client=c,
        )
        c.upload_dataset.assert_called_once_with(
            path="data.csv", name="my-ds", dataset_type="tabular", format="csv",
            description="test", visibility="public", license="mit", progress_cb=None,
        )
        assert out["id"] == "ds1"

    def test_upload_passes_progress_cb(self):
        cb = MagicMock()
        c = _client(upload_dataset=MagicMock(return_value={"id": "ds1"}))
        datasets_upload.upload(
            "data.csv", name="ds", dataset_type="tabular", format="csv",
            progress_cb=cb, client=c,
        )
        assert c.upload_dataset.call_args.kwargs["progress_cb"] is cb


class TestUploadFromUrl:
    def test_returns_lro(self):
        c = _client(upload_dataset_from_url=MagicMock(
            return_value={"task_id": "t1", "status": "pending"},
        ))
        op = datasets_upload.upload_from_url(
            "https://example.com/data.parquet",
            name="remote-ds", dataset_type="tabular", format="parquet",
            client=c,
        )
        assert isinstance(op, LongRunningOperation)
        assert op.initial()["task_id"] == "t1"
        c.upload_dataset_from_url.assert_called_once_with(
            url="https://example.com/data.parquet",
            name="remote-ds", dataset_type="tabular", format="parquet",
            description=None, visibility="private",
        )

    def test_lro_polls_task_status(self):
        c = _client(
            upload_dataset_from_url=MagicMock(
                return_value={"task_id": "t1", "status": "pending"},
            ),
            get_dataset_task_status=MagicMock(
                return_value={"status": "completed", "dataset_id": "ds1"},
            ),
        )
        op = datasets_upload.upload_from_url(
            "https://x.com/d.csv", name="ds", dataset_type="tabular", format="csv",
            client=c,
        )
        # Manually invoke the poll to verify it calls the right method
        result = op._poll()
        c.get_dataset_task_status.assert_called_once_with("t1")
        assert result["status"] == "completed"


class TestErrorPropagation:
    def test_upload_propagates_upload_error(self):
        c = _client()
        c.upload_dataset.side_effect = UploadError("too large")
        with pytest.raises(UploadError):
            datasets_upload.upload(
                "data.csv", name="ds", dataset_type="tabular", format="csv",
                client=c,
            )

    def test_upload_from_url_propagates_upload_error(self):
        c = _client()
        c.upload_dataset_from_url.side_effect = UploadError("bad url")
        with pytest.raises(UploadError):
            datasets_upload.upload_from_url(
                "https://x.com/d.csv", name="ds", dataset_type="tabular",
                format="csv", client=c,
            )
