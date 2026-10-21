"""Hub files after upload: wait for the platform's inspection, remove a file, link a design.

Every uploaded file is inspected by the platform before anyone else can see it or its version
can be published; ``wait_for_inspection`` polls the model's files until none is pending. A
platform without inspection reports no state, which reads as done.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import time
from typing import Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    HubError,
    HubModelNotFoundError,
    InspectionTimeoutError,
    UploadError,
)
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


_CONVERTED_SUFFIXES = frozenset({"pt", "pth", "h5"})


def stored_name(file_path: str) -> str:
    """The name the hub stores a local file under.

    A PyTorch ``.pt``/``.pth`` or HDF5 ``.h5`` file is converted by the platform and stored as
    ``<stem>.safetensors`` (``model.weights.h5`` becomes ``model.weights.safetensors``);
    every other file keeps its base name. Two local files with the same stored name are one
    file to the hub, which is why ``publish`` refuses them up front.
    """
    name = Path(file_path).name
    stem, dot, suffix = name.rpartition(".")
    if dot and suffix.lower() in _CONVERTED_SUFFIXES:
        return f"{stem}.safetensors"
    return name


def replace_refused_copy(
    resolved: DagnamClient, model_id: str, version_id: Optional[str], file_path: str
) -> bool:
    """Remove the server's refused file of this name, so a resumed upload can try again."""
    name = stored_name(file_path).casefold()
    for entry in listed_files(resolved, model_id, version_id):
        file_name = entry.get("file_name")
        if (
            entry.get("scan_status") == "rejected"
            and isinstance(file_name, str)
            and file_name.casefold() == name
        ):
            resolved.delete_hub_model_file(model_id, str(entry["id"]))
            return True
    return False


def _refusal(entry: JsonObject) -> str:
    rejection = entry.get("rejection")
    reason = rejection if isinstance(rejection, dict) else {}
    return (
        f"{entry.get('file_name')} was refused: {reason.get('message', 'no reason given')} "
        f"({reason.get('code', 'refused')})"
    )


def await_clean_files(
    resolved: DagnamClient,
    model_id: str,
    version_id: Optional[str],
    files: Sequence[str],
    on_file_progress: Optional[Callable[[str, int, int, str], object]],
    advice: str,
) -> None:
    """Wait until the platform has inspected every file of the version, then remove the refused ones.

    Every file counts, not only this run's: a resumed publish skips files an earlier run
    uploaded. Raises ``UploadError`` naming each refused file (they are already removed). A wait
    that runs out raises ``InspectionTimeoutError`` and a failed list or delete call raises
    ``UploadError``; both name the draft and say ``advice``, because the draft is kept.
    """
    subject = (
        f"hub model {model_id}"
        if version_id is None
        else f"version {version_id} of hub model {model_id}"
    )
    refused: list[str] = []
    local = {stored_name(path).casefold(): (i, path) for i, path in enumerate(files, start=1)}
    try:
        for entry in wait_for_inspection(model_id, version_id=version_id, client=resolved):
            if entry.get("scan_status") != "rejected":
                continue
            resolved.delete_hub_model_file(model_id, str(entry["id"]))
            refused.append(_refusal(entry))
            if on_file_progress is not None:
                index, path = local.get(
                    str(entry.get("file_name")).casefold(), (0, str(entry.get("file_name")))
                )
                on_file_progress(path, index, len(files), "refused")
    except InspectionTimeoutError as exc:
        raise InspectionTimeoutError(
            f"Publish of {subject} halted: the files are still being inspected after the wait "
            f"ran out. The draft is kept: wait, then {advice}."
        ) from exc
    except (APIError, HubError, HubModelNotFoundError) as exc:
        raise UploadError(
            f"Publish of {subject} halted: could not check or clear the files ({exc}). "
            f"The draft is kept: {advice}."
        ) from exc
    if refused:
        raise UploadError(
            f"Publish of {subject} halted: {'; '.join(refused)}. The refused file(s) were removed "
            f"from the draft; fix them and {advice}."
        )


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


__all__ = [
    "InspectionTimeoutError",
    "await_clean_files",
    "delete_file",
    "link_design",
    "replace_refused_copy",
    "stored_name",
    "wait_for_inspection",
]
