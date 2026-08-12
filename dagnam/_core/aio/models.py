"""Async model registry client methods."""

from __future__ import annotations

from pathlib import Path

import httpx

from dagnam._core.aio.base import BaseAsyncDagnamClient
from dagnam._core.client.base import scrub_secret_params
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_model,
    response_json_value,
    safe_response_text,
)
from dagnam._core.exceptions import APIError, AuthError, ModelNotFoundError, ResponseError
from dagnam._types import (
    JsonArray,
    JsonObject,
    JsonValue,
    QueryParams,
    QueryValue,
    ensure_json_array,
    ensure_json_object,
)


class AsyncModelsMixin(BaseAsyncDagnamClient):
    """Async model registry resource methods."""

    async def _registry_req(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        idempotent: bool = False,
    ) -> JsonValue | str | None:
        """Issue an authenticated request against a registry route.

        ``idempotent=True`` mints an ``Idempotency-Key`` so a transient
        failure on a create POST (entry/version/artifact-initiate) retries
        into a server-side replay instead of orphaning a duplicate draft.
        """
        resp = await self._request(
            method,
            path,
            params=params,
            json=json_body,
            raise_for=lambda r: raise_for_model(r, model_id),
            idempotent=idempotent,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    # ---------------------------------------------------------------- entries

    async def create_model_entry(self, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._registry_req("POST", "/api/v1/models", json_body=payload, idempotent=True)
        )

    async def get_model_entry(self, model_id: str) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "GET", f"/api/v1/models/{quote_path_segment(model_id)}", model_id=model_id
            )
        )

    async def list_model_entries(self, **filter_params: QueryValue) -> JsonArray:
        return ensure_json_array(
            await self._registry_req("GET", "/api/v1/models", params=filter_params)
        )

    async def update_model_entry(self, model_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "PATCH",
                f"/api/v1/models/{quote_path_segment(model_id)}",
                model_id=model_id,
                json_body=payload,
            )
        )

    async def delete_model_entry(self, model_id: str) -> None:
        await self._registry_req(
            "DELETE", f"/api/v1/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    # --------------------------------------------------------------- versions

    async def create_model_version(self, model_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "POST",
                f"/api/v1/models/{quote_path_segment(model_id)}/versions",
                model_id=model_id,
                json_body=payload,
                idempotent=True,
            )
        )

    async def list_model_versions(self, model_id: str) -> JsonArray:
        return ensure_json_array(
            await self._registry_req(
                "GET", f"/api/v1/models/{quote_path_segment(model_id)}/versions", model_id=model_id
            )
        )

    async def get_model_version(self, version_id: str) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "GET",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}",
                model_id=version_id,
            )
        )

    async def get_model_version_lineage(self, version_id: str) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "GET",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/lineage",
                model_id=version_id,
            )
        )

    # -------------------------------------------------------------- artifacts

    async def initiate_model_artifact(self, version_id: str, payload: JsonObject) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "POST",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/artifacts:initiate",
                model_id=version_id,
                json_body=payload,
                idempotent=True,
            )
        )

    async def complete_model_artifact(
        self, version_id: str, artifact_id: str, payload: JsonObject
    ) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "POST",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}"
                f"/artifacts/{quote_path_segment(artifact_id)}/complete",
                model_id=version_id,
                json_body=payload,
            )
        )

    async def finalize_model_version(self, version_id: str) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "POST",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/finalize",
                model_id=version_id,
            )
        )

    async def get_task_contract(self, key: str, version: str) -> JsonObject:
        return ensure_json_object(
            await self._registry_req(
                "GET",
                f"/api/v1/task-contracts/{quote_path_segment(key)}"
                f"/versions/{quote_path_segment(version)}",
            )
        )

    async def list_model_version_artifacts(self, version_id: str) -> JsonArray:
        return ensure_json_array(
            await self._registry_req(
                "GET",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/artifacts",
                model_id=version_id,
            )
        )

    async def upload_model_artifact_direct(self, upload_url: str, file_path: Path) -> None:
        """Async mirror of ``ModelsClientMixin.upload_model_artifact_direct``.

        Unlike the sync version, ``httpx`` sets its own multipart boundary
        Content-Type even when routed through ``self._request`` (see
        ``AsyncHubMixin.upload_model_file``), so no raw-client workaround is
        needed here.
        """
        with file_path.open("rb") as fh:
            resp = await self._request("POST", upload_url, files={"file": (file_path.name, fh)})
        raise_for_model(resp)

    async def _raise_for_model_artifact(self, resp: httpx.Response, artifact_id: str) -> None:
        await resp.aread()  # populate the body for the error message
        code = resp.status_code
        if code == 401:
            raise AuthError("Authentication failed: invalid or expired API key")
        if code == 404:
            raise ModelNotFoundError(artifact_id)
        raise APIError(code, safe_response_text(resp))

    async def download_model_artifact(
        self, version_id: str, artifact_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        """Download one artifact to ``dest_path``.

        Async mirror of ``ModelsClientMixin.download_model_artifact_stream``
        (itself mirroring ``AsyncCheckpointsMixin.download_checkpoint``): follows
        a 307/308 redirect to a presigned object-storage URL (fetched WITHOUT
        the API key) on ``STORAGE_BACKEND=s3``, or streams the body directly on
        ``local``. Both paths stream straight to disk. Returns
        ``(dest_path, expected_sha256)`` — the caller must verify the digest.
        """
        path = (
            f"/api/v1/model-versions/{quote_path_segment(version_id)}"
            f"/artifacts/{quote_path_segment(artifact_id)}/download"
        )
        url = f"{self.api_url}{path}"
        dest = Path(dest_path)
        location: str | None = None
        expected_checksum: str | None = None
        try:
            async with self._client.stream("GET", url, headers=self._headers()) as resp:
                expected_checksum = resp.headers.get("x-checksum-sha256")
                if resp.is_redirect and resp.headers.get("location"):
                    location = resp.headers["location"]
                else:
                    if not resp.is_success:
                        await self._raise_for_model_artifact(resp, artifact_id)
                    await self._stream_response_to_file(resp, dest)
                    return dest, expected_checksum

            # Presigned redirect: stream the object-storage URL without auth
            # (the signature is in the query string, never a forwarded header).
            async with self._client.stream("GET", location) as resp:
                expected_checksum = resp.headers.get("x-checksum-sha256") or expected_checksum
                if not resp.is_success:
                    await self._raise_for_model_artifact(resp, artifact_id)
                await self._stream_response_to_file(resp, dest)
        except httpx.ConnectError as exc:
            raise APIError(0, f"Connection failed: {scrub_secret_params(str(exc))}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(0, f"Request timed out: {scrub_secret_params(str(exc))}") from exc
        return dest, expected_checksum
