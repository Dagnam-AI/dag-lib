"""Synchronous model registry client methods."""

from __future__ import annotations

from pathlib import Path

from dagnam._core.client.base import (
    ALLOW_REDIRECTS,
    DEFAULT_TIMEOUT,
    APIError,
    BaseDagnamClient,
    is_redirect_response,
    is_success_response,
    requests,
    safe_error_body_from_response,
)
from dagnam._core.client.common import (
    quote_path_segment,
    raise_for_model,
    requests_query_params,
    response_json_value,
)
from dagnam._core.exceptions import AuthError, ModelNotFoundError, ResponseError
from dagnam._types import JsonArray, JsonObject, JsonValue, QueryParams, QueryValue


class ModelsClientMixin(BaseDagnamClient):
    """Model registry resource methods for DagnamClient."""

    def _registry_request(
        self,
        method: str,
        path: str,
        *,
        model_id: str | None = None,
        params: QueryParams | None = None,
        json_body: JsonValue = None,
        timeout: int = DEFAULT_TIMEOUT,
        idempotent: bool = False,
    ) -> JsonValue | str | None:
        """Issue an authenticated request against a registry route.

        ``idempotent=True`` mints an ``Idempotency-Key`` so a transient
        failure on a create POST (entry/version/artifact-initiate) retries
        into a server-side replay instead of orphaning a duplicate draft.
        """
        url = f"{self.api_url}{path}"
        resp = self._request(
            method,
            url,
            raise_for=lambda r: raise_for_model(r, model_id),
            params=requests_query_params(params),
            json=json_body,
            timeout=timeout,
            allow_redirects=ALLOW_REDIRECTS,
            idempotent=idempotent,
        )
        if not resp.content:
            return None
        try:
            return response_json_value(resp)
        except ResponseError:
            return resp.text

    # ---------------------------------------------------------------- entries

    def create_model_entry(self, payload: JsonObject) -> JsonObject:
        return self._expect_object(
            self._registry_request("POST", "/api/v1/models", json_body=payload, idempotent=True)
        )

    def get_model_entry(self, model_id: str) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "GET", f"/api/v1/models/{quote_path_segment(model_id)}", model_id=model_id
            )
        )

    def list_model_entries(self, **filter_params: QueryValue) -> JsonArray:
        return self._expect_array(
            self._registry_request("GET", "/api/v1/models", params=filter_params)
        )

    def update_model_entry(self, model_id: str, payload: JsonObject) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "PATCH",
                f"/api/v1/models/{quote_path_segment(model_id)}",
                model_id=model_id,
                json_body=payload,
            )
        )

    def delete_model_entry(self, model_id: str) -> None:
        self._registry_request(
            "DELETE", f"/api/v1/models/{quote_path_segment(model_id)}", model_id=model_id
        )

    # --------------------------------------------------------------- versions

    def create_model_version(self, model_id: str, payload: JsonObject) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "POST",
                f"/api/v1/models/{quote_path_segment(model_id)}/versions",
                model_id=model_id,
                json_body=payload,
                idempotent=True,
            )
        )

    def list_model_versions(self, model_id: str) -> JsonArray:
        return self._expect_array(
            self._registry_request(
                "GET", f"/api/v1/models/{quote_path_segment(model_id)}/versions", model_id=model_id
            )
        )

    def get_model_version(self, version_id: str) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "GET",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}",
                model_id=version_id,
            )
        )

    def get_model_version_lineage(self, version_id: str) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "GET",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/lineage",
                model_id=version_id,
            )
        )

    # -------------------------------------------------------------- artifacts

    def initiate_model_artifact(self, version_id: str, payload: JsonObject) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "POST",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/artifacts:initiate",
                model_id=version_id,
                json_body=payload,
                idempotent=True,
            )
        )

    def complete_model_artifact(
        self, version_id: str, artifact_id: str, payload: JsonObject
    ) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "POST",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}"
                f"/artifacts/{quote_path_segment(artifact_id)}/complete",
                model_id=version_id,
                json_body=payload,
            )
        )

    def finalize_model_version(self, version_id: str) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "POST",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/finalize",
                model_id=version_id,
            )
        )

    def get_task_contract(self, key: str, version: str) -> JsonObject:
        return self._expect_object(
            self._registry_request(
                "GET",
                f"/api/v1/task-contracts/{quote_path_segment(key)}"
                f"/versions/{quote_path_segment(version)}",
            )
        )

    def list_model_version_artifacts(self, version_id: str) -> JsonArray:
        return self._expect_array(
            self._registry_request(
                "GET",
                f"/api/v1/model-versions/{quote_path_segment(version_id)}/artifacts",
                model_id=version_id,
            )
        )

    def upload_model_artifact_direct(self, upload_url: str, file_path: Path) -> None:
        """POST a file directly to a local-backend upload endpoint.

        The URL comes from ``initiate_model_artifact``. Uses raw
        ``requests.post`` (not ``self._request``), matching
        ``HubClientMixin.upload_model_file`` exactly: ``requests`` must set its
        own multipart Content-Type boundary header, which a JSON-oriented
        ``self._request`` would override. On ``STORAGE_BACKEND=s3`` the caller
        never reaches this method — it PUTs straight to the presigned URL
        ``initiate_model_artifact`` returned.
        """
        path = Path(file_path)
        url = f"{self.api_url}{upload_url}"
        try:
            with path.open("rb") as fh:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    files={"file": (path.name, fh)},
                    timeout=DEFAULT_TIMEOUT,
                    allow_redirects=ALLOW_REDIRECTS,
                )
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise APIError(0, f"Request timed out: {exc}") from exc
        raise_for_model(resp)

    def download_model_artifact_stream(
        self, version_id: str, artifact_id: str, dest_path: Path
    ) -> tuple[Path, str | None]:
        """Stream-download one artifact to ``dest_path``.

        Mirrors ``CheckpointsClientMixin.download_checkpoint_stream``: follows a
        307/308 redirect to a presigned object-storage URL (fetched with no
        ``Authorization`` header — the URL itself is the credential) on
        ``STORAGE_BACKEND=s3``, or streams the response body directly on
        ``local``. Returns ``(dest_path, expected_sha256)`` — the caller must
        verify the checksum.
        """
        url = (
            f"{self.api_url}/api/v1/model-versions/{quote_path_segment(version_id)}"
            f"/artifacts/{quote_path_segment(artifact_id)}/download"
        )
        resp = self._get_stream(url)
        expected_checksum = resp.headers.get("X-Checksum-SHA256")

        if is_redirect_response(resp):
            location = resp.headers["Location"]
            resp.close()
            resp = self._get_stream_no_auth(location)
            expected_checksum = resp.headers.get("X-Checksum-SHA256") or expected_checksum

        if not is_success_response(resp):
            code = resp.status_code
            if code == 401:
                raise AuthError("Authentication failed: invalid or expired API key")
            if code == 404:
                raise ModelNotFoundError(artifact_id)
            raise APIError(code, safe_error_body_from_response(resp))

        written = self._stream_response_to_file(resp, Path(dest_path))
        return written, expected_checksum
