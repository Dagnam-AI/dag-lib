"""Model Hub publishing: create -> upload files -> (version) -> finalize, as one call.

Kept apart from :mod:`dagnam.resources.hub` (the per-route functions) so each file stays
readable; ``dagnam.hub.publish`` is re-exported from there.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import (
    APIError,
    HubError,
    HubModelNotFoundError,
    PayloadTooLargeError,
    UploadError,
)
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject, JsonValue

_RETRYABLE_STATUS = frozenset({0, 408, 429})


def _retryable(exc: Exception) -> bool:
    """Only a transport failure (0), a timeout (408), a rate limit (429) or a 5xx can succeed again.

    A rejected file, a conflict, a request above the size ceiling (413) and a local file that
    changed mid-send (``OSError``) answer the same way on every attempt.
    """
    return isinstance(exc, APIError) and (
        exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    )


def create_payload(
    *,
    name: str,
    description: str,
    task_type: str,
    framework: str,
    license: str,
    visibility: str,
    tags: Optional[list[str]],
    metadata: Optional[JsonObject],
) -> JsonObject:
    """The body of ``POST /hub/models``, shared by ``hub.create`` and :func:`publish`."""
    payload: JsonObject = {
        "name": name,
        "description": description,
        "task_type": task_type,
        "framework": framework,
        "license": license,
        "visibility": visibility,
    }
    if tags is not None:
        payload["tags"] = [str(tag) for tag in tags]
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def finalize(
    model_id: str | UUID,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Publish a draft model: ``POST /hub/models/{id}/finalize``.

    Raises ``HubModelNotFoundError`` when the platform answers 404 and ``APIError`` for any
    other refusal; the model stays a draft in both cases.
    """
    resolved = resolve_client(client, api_key, api_url)
    return resolved.finalize_hub_model(str(model_id))


def publish(
    *,
    name: str,
    description: str,
    task_type: str,
    framework: str,
    files: Sequence[str],
    version: Optional[str] = None,
    changelog: Optional[str] = None,
    license: str = "mit",
    visibility: str = "public",
    tags: Optional[list[str]] = None,
    metadata: Optional[JsonObject] = None,
    max_retries_per_file: int = 2,
    on_file_progress: Optional[Callable[[str, int, int, str], object]] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Publish a model to the hub: create -> upload files -> finalize.

    Follows the draft-to-finalize publish contract: the model record is created
    first, every file is uploaded (with ``max_retries_per_file`` retries per
    file), an optional version is recorded, and a finalize call flips the
    model live.

    ``on_file_progress(path, index, total, state)`` receives per-file states:
    ``uploading`` -> (``retrying`` ...) -> ``uploaded`` | ``failed``.

    Raises ``FileNotFoundError`` before any network call if a local file is
    missing, and ``UploadError`` if a file still fails after retries -- the
    created model and already-uploaded files are left in place so the publish
    can be resumed with ``upload_file`` + a re-run.
    """
    for file_path in files:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"No such file: {file_path}")

    resolved = resolve_client(client, api_key, api_url)
    model = resolved.create_hub_model(
        create_payload(
            name=name,
            description=description,
            task_type=task_type,
            framework=framework,
            license=license,
            visibility=visibility,
            tags=tags,
            metadata=metadata,
        )
    )
    model_id = str(model["id"])

    total = len(files)
    uploaded: list[JsonValue] = []
    for index, file_path in enumerate(files, start=1):
        if on_file_progress is not None:
            on_file_progress(file_path, index, total, "uploading")
        attempts = 0
        while True:
            try:
                uploaded.append(resolved.upload_model_file(model_id, file_path))
                if on_file_progress is not None:
                    on_file_progress(file_path, index, total, "uploaded")
                break
            except (APIError, HubError, UploadError, PayloadTooLargeError, OSError) as exc:
                attempts += 1
                if attempts > max_retries_per_file or not _retryable(exc):
                    if on_file_progress is not None:
                        on_file_progress(file_path, index, total, "failed")
                    done = ", ".join(str(Path(p).name) for p in files[: index - 1]) or "none"
                    raise UploadError(
                        f"Publish of hub model {model_id} halted: {file_path} failed "
                        f"after {attempts} attempt(s) ({exc}). Uploaded so far: {done}. "
                        f"Fix the issue, then resume with hub.upload_file({model_id!r}, ...)."
                    ) from exc
                if on_file_progress is not None:
                    on_file_progress(file_path, index, total, "retrying")

    version_record: JsonValue = None
    if version is not None:
        version_payload: JsonObject = {"version": version}
        if changelog is not None:
            version_payload["changelog"] = changelog
        version_record = resolved.create_hub_model_version(model_id, version_payload)

    try:
        model = finalize(model_id, client=resolved)
    except (HubModelNotFoundError, APIError) as exc:
        code = exc.status_code if isinstance(exc, APIError) else 404
        if code not in (404, 405):
            raise
        raise HubError(
            f"Hub model {model_id} has its files but is still a draft: the platform answered "
            f"HTTP {code} to the finalize call. Retry that step with hub.finalize({model_id!r})."
        ) from exc

    return {"model": model, "files": uploaded, "version": version_record, "finalized": True}


__all__ = ["create_payload", "finalize", "publish"]
