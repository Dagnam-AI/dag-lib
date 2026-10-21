"""Hub files after upload: wait for the platform's inspection, remove a file, link a design.

Every uploaded file is inspected by the platform before anyone else can see it or its version
can be published; ``wait_for_inspection`` polls the model's files until none is pending. A
platform without inspection reports no state, which reads as done.
"""

from __future__ import annotations

import time
from typing import Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import InspectionTimeoutError
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject


def listed_files(
    client: DagnamClient, model_id: str, version_id: Optional[str] = None
) -> list[JsonObject]:
    """The file objects of a version (the page's default list without ``version_id``)."""
    listing = client.list_hub_model_files(model_id, version_id=version_id).get("files")
    return (
        [entry for entry in listing if isinstance(entry, dict)] if isinstance(listing, list) else []
    )


def wait_for_inspection(
    model_id: str | UUID,
    *,
    version_id: str | UUID | None = None,
    timeout: float = 600.0,
    poll: float = 3.0,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> list[JsonObject]:
    """Poll until no file of the version is pending, then return every file.

    ``version_id`` defaults to the page's default list (the draft of a draft page). All files
    count, not only this run's: a resumed publish skips files an earlier run uploaded, which may
    still be pending. Raises ``InspectionTimeoutError`` after ``timeout`` seconds.
    """
    resolved = resolve_client(client, api_key, api_url)
    version = None if version_id is None else str(version_id)
    started = time.monotonic()
    while True:
        files = listed_files(resolved, str(model_id), version)
        if not any(entry.get("scan_status") == "pending" for entry in files):
            return files
        if time.monotonic() - started > timeout:
            raise InspectionTimeoutError(
                f"Hub model {model_id}: files are still being inspected after {timeout:.0f} s. "
                "Wait, then publish the draft with hub.finalize(model_id), or with "
                "hub.finalize_version(model_id, version_id) for a new version."
            )
        time.sleep(poll)


def delete_file(
    model_id: str | UUID,
    file_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> None:
    """Remove one file: a refused file, or any file of a draft version."""
    resolve_client(client, api_key, api_url).delete_hub_model_file(str(model_id), str(file_id))


def link_design(
    model_id: str | UUID,
    version_id: str | UUID,
    *,
    project_id: Optional[str] = None,
    architecture_version_id: Optional[str] = None,
    architecture: Optional[JsonObject] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Give a weights-only version its design: from a project you own, or a Studio design object.

    The platform accepts it once, when the design needs between half of and all the floating
    point values the weights hold.
    """
    if (project_id is None) == (architecture is None):
        raise ValueError("Pass exactly one of project_id or architecture")
    if architecture_version_id is not None and project_id is None:
        raise ValueError("architecture_version_id goes with project_id")
    body: JsonObject = (
        {"architecture": architecture} if architecture is not None else {"project_id": project_id}
    )
    if architecture_version_id is not None:
        body["architecture_version_id"] = architecture_version_id
    return resolve_client(client, api_key, api_url).link_hub_model_design(
        str(model_id), str(version_id), body
    )


__all__ = ["InspectionTimeoutError", "delete_file", "link_design", "wait_for_inspection"]
