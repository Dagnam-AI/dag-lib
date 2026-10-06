"""The audit id a dataset create carries, and the audit filter a listing is asked for."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

    from dagnam._core.client import DagnamClient

API = "https://api.test"


def test_list_datasets_can_ask_for_one_audits_datasets(
    client: DagnamClient, rmock: RequestsMocker
) -> None:
    rmock.get(f"{API}/api/v1/datasets/browse", json=[])
    client.list_datasets(audit_id="audit-1")
    assert rmock.last_request.qs == {"type": ["all"], "audit_id": ["audit-1"]}


def test_upload_dataset_names_the_audit_so_the_platform_tags_it_at_creation(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    f = tmp_path / "data.jsonl"
    f.write_text("{}")
    rmock.post(f"{API}/api/v1/datasets/", json={"id": "ds1"})
    client.upload_dataset(f, name="x", dataset_type="text", format="json", audit_id="audit-1")
    assert 'name="audit_id"' in str(rmock.last_request.text)
    client.upload_dataset(f, name="x", dataset_type="text", format="json")
    assert 'name="audit_id"' not in str(rmock.last_request.text)
