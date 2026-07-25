"""Unit tests for dagnam.datasets upload helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dagnam import datasets as datasets_upload
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import LROFailedError, UploadError
from dagnam._core.lro import LongRunningOperation


def _client(**overrides: object) -> MagicMock:
    client = MagicMock(spec=DagnamClient)
    client.configure_mock(**overrides)
    return client


class TestUpload:
    def test_upload_delegates(self) -> None:
        c = _client(upload_dataset=MagicMock(return_value={"id": "ds1"}))
        out = datasets_upload.upload(
            "data.csv",
            name="my-ds",
            dataset_type="tabular",
            format="csv",
            description="test",
            visibility="public",
            license="mit",
            client=c,
        )
        c.upload_dataset.assert_called_once_with(
            file_path="data.csv",
            name="my-ds",
            dataset_type="tabular",
            format="csv",
            description="test",
            visibility="public",
            license="mit",
            progress_cb=None,
        )
        assert out["id"] == "ds1"

    def test_upload_passes_progress_cb(self) -> None:
        cb = MagicMock()
        c = _client(upload_dataset=MagicMock(return_value={"id": "ds1"}))
        datasets_upload.upload(
            "data.csv",
            name="ds",
            dataset_type="tabular",
            format="csv",
            progress_cb=cb,
            client=c,
        )
        assert c.upload_dataset.call_args.kwargs["progress_cb"] is cb


class TestUploadFromUrl:
    def test_returns_lro(self) -> None:
        c = _client(
            upload_dataset_from_url=MagicMock(
                return_value={"task_id": "t1", "status": "pending"},
            )
        )
        op = datasets_upload.upload_from_url(
            "https://example.com/data.parquet",
            name="remote-ds",
            dataset_type="tabular",
            format="parquet",
            client=c,
        )
        assert isinstance(op, LongRunningOperation)
        initial = op.initial()
        assert initial is not None
        assert initial["task_id"] == "t1"
        c.upload_dataset_from_url.assert_called_once_with(
            url="https://example.com/data.parquet",
            name="remote-ds",
            dataset_type="tabular",
            format="parquet",
            description=None,
            visibility="private",
        )

    def test_missing_string_task_id_raises(self) -> None:
        # Response without a string task_id must be rejected before building the LRO.
        c = _client(upload_dataset_from_url=MagicMock(return_value={"status": "pending"}))
        with pytest.raises(ValueError, match="did not include a string task_id"):
            datasets_upload.upload_from_url(
                "https://x.com/d.csv",
                name="ds",
                dataset_type="tabular",
                format="csv",
                client=c,
            )

    def test_lro_polls_task_status(self) -> None:
        c = _client(
            upload_dataset_from_url=MagicMock(
                return_value={"task_id": "t1", "status": "pending"},
            ),
            get_dataset_task_status=MagicMock(
                return_value={"status": "completed", "dataset_id": "ds1"},
            ),
        )
        op = datasets_upload.upload_from_url(
            "https://x.com/d.csv",
            name="ds",
            dataset_type="tabular",
            format="csv",
            client=c,
        )
        # Manually invoke the poll to verify it calls the right method
        result = op._poll()
        c.get_dataset_task_status.assert_called_once_with("t1")
        assert result["status"] == "completed"

    def _op(self, poll_payload: dict[str, object]) -> LongRunningOperation:
        c = _client(
            upload_dataset_from_url=MagicMock(return_value={"task_id": "t1", "status": "pending"}),
            get_dataset_task_status=MagicMock(return_value=poll_payload),
        )
        return datasets_upload.upload_from_url(
            "https://x.com/d.csv",
            name="ds",
            dataset_type="tabular",
            format="csv",
            client=c,
        )

    @pytest.mark.parametrize("state", ["SUCCESS", "success", "completed", "ready"])
    def test_lro_accepts_every_success_spelling(self, state: str) -> None:
        # The task-status endpoint reports the raw queue status ("SUCCESS")
        # under `status`; the lower-cased rendering is what `state` carries.
        op = self._op({"status": state, "dataset_id": "ds1"})
        assert op.wait(timeout=1, sleep=lambda _d: None).result()["dataset_id"] == "ds1"

    @pytest.mark.parametrize("state", ["FAILURE", "failure", "failed", "REVOKED", "cancelled"])
    def test_lro_accepts_every_failure_spelling(self, state: str) -> None:
        op = self._op({"status": state, "error": "download failed"})
        op.wait(timeout=1, sleep=lambda _d: None)
        with pytest.raises(LROFailedError, match="download failed"):
            op.result()

    def test_lro_reads_the_error_field(self) -> None:
        # The server sends the failure detail as `error`; `error_message` stays
        # supported as a fallback for any payload that still uses it.
        op = self._op({"status": "FAILURE", "error": "bad url"})
        op.wait(timeout=1, sleep=lambda _d: None)
        with pytest.raises(LROFailedError, match="bad url"):
            op.result()

    def test_lro_falls_back_to_error_message(self) -> None:
        op = self._op({"status": "FAILURE", "error_message": "legacy detail"})
        op.wait(timeout=1, sleep=lambda _d: None)
        with pytest.raises(LROFailedError, match="legacy detail"):
            op.result()


class TestList:
    def test_list_delegates_with_defaults(self) -> None:
        c = _client(list_datasets=MagicMock(return_value=[{"id": "ds1"}]))
        out = datasets_upload.list(client=c)
        c.list_datasets.assert_called_once_with(type="all", search=None)
        assert out == [{"id": "ds1"}]

    def test_list_passes_filters(self) -> None:
        c = _client(list_datasets=MagicMock(return_value=[]))
        out = datasets_upload.list(type="tabular", search="iris", client=c)
        c.list_datasets.assert_called_once_with(type="tabular", search="iris")
        assert out == []


class TestListSystem:
    def test_list_system_delegates(self) -> None:
        c = _client(list_system_datasets=MagicMock(return_value=[{"id": "mnist"}]))
        out = datasets_upload.list_system(client=c)
        c.list_system_datasets.assert_called_once_with()
        assert out == [{"id": "mnist"}]


class TestGet:
    def test_get_delegates(self) -> None:
        c = _client(get_dataset_meta=MagicMock(return_value={"id": "ds1", "name": "n"}))
        out = datasets_upload.get("ds1", client=c)
        c.get_dataset_meta.assert_called_once_with("ds1", version=None)
        assert out["name"] == "n"

    def test_get_passes_version(self) -> None:
        c = _client(get_dataset_meta=MagicMock(return_value={"id": "ds1"}))
        datasets_upload.get("ds1", version="v2", client=c)
        c.get_dataset_meta.assert_called_once_with("ds1", version="v2")


class TestErrorPropagation:
    def test_upload_propagates_uploaderror(self) -> None:
        c = _client()
        c.upload_dataset.side_effect = UploadError("too large")
        with pytest.raises(UploadError):
            datasets_upload.upload(
                "data.csv",
                name="ds",
                dataset_type="tabular",
                format="csv",
                client=c,
            )

    def test_upload_from_url_propagates_uploaderror(self) -> None:
        c = _client()
        c.upload_dataset_from_url.side_effect = UploadError("bad url")
        with pytest.raises(UploadError):
            datasets_upload.upload_from_url(
                "https://x.com/d.csv",
                name="ds",
                dataset_type="tabular",
                format="csv",
                client=c,
            )
