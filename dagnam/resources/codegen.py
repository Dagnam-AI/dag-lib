"""Code generation — sync SDK surface.

Wraps the ``/api/v1/codegen/*`` routes on top of
:class:`dagnam.client.DagnamClient`.  The ``generate`` function supports
an ``async_mode`` flag that returns a
:class:`~dagnam.lro.LongRunningOperation` polling until the generation
task reaches ``completed`` or ``failed``.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Optional, Union
import zipfile

from dagnam._core.client import DagnamClient
from dagnam._core.lro import LongRunningOperation
from dagnam._core.resolver import resolve_client
from dagnam._types import JsonObject
from dagnam.data.loaders.media import safe_extract_zip


def generate(
    project_id: str,
    *,
    framework: str = "pytorch",
    version_id: Optional[str] = None,
    async_mode: bool = False,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Union[JsonObject, LongRunningOperation]:
    """Generate code for a project.

    When *async_mode* is ``True``, return an LRO that polls
    ``get_code_status`` until the task reaches ``completed`` or ``failed``.

    >>> result = dagnam.codegen.generate("proj_abc")
    >>> op = dagnam.codegen.generate("proj_abc", async_mode=True)
    >>> result = op.wait(timeout=300).result()
    """
    resolved = resolve_client(client, api_key, api_url)
    resp = resolved.generate_code(
        project_id,
        framework=framework,
        version_id=version_id,
        async_mode=async_mode,
    )
    if not async_mode:
        return resp
    task_id_value = resp.get("task_id")
    if not isinstance(task_id_value, str):
        raise ValueError("Code generation response did not include a string task_id")
    return LongRunningOperation(
        poll=lambda: resolved.get_code_status(project_id, task_id_value),
        success_states={"completed"},
        failure_states={"failed"},
        state_key="status",
        name=f"codegen.generate({project_id})",
        initial=resp,
    )


def preview(
    project_id: str,
    *,
    framework: str = "pytorch",
    version_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject | str | None:
    """Preview generated code without persisting it."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.preview_code(
        project_id,
        framework=framework,
        version_id=version_id,
    )


def validate(
    project_id: str,
    *,
    version_id: Optional[str] = None,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Validate a project's code generation readiness."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.validate_code(project_id, version_id=version_id)


def download(
    project_id: str,
    *,
    framework: str = "pytorch",
    version_id: Optional[str] = None,
    dest: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Union[Path, bytes]:
    """Download generated code.

    If *dest* is an existing directory, the generated code archive is
    downloaded to a temporary file and extracted into *dest*, which is then
    returned.  If *dest* is a file path, the raw archive is streamed there and
    the :class:`~pathlib.Path` is returned.  Otherwise the raw bytes are
    returned.  *show_progress* only applies when streaming to *dest*.
    """
    resolved = resolve_client(client, api_key, api_url)
    if dest is not None:
        dest = Path(dest)
        if dest.is_dir():
            # Convenience: download the archive to a temp file and extract into dest.
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                archive_path = Path(tmp.name)
            try:
                downloaded = resolved.download_code(
                    project_id,
                    framework=framework,
                    version_id=version_id,
                    dest_path=archive_path,
                    show_progress=show_progress,
                )
                if not isinstance(downloaded, Path):
                    raise TypeError("Expected a download path for the temporary archive")
                with zipfile.ZipFile(downloaded) as zf:
                    # Hardened extraction: a tampered archive with traversal /
                    # absolute / symlink members is refused, never written
                    # outside dest (zip-slip).
                    safe_extract_zip(zf, dest)
            finally:
                archive_path.unlink(missing_ok=True)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
    data = resolved.download_code(
        project_id,
        framework=framework,
        version_id=version_id,
        dest_path=dest,
        show_progress=show_progress,
    )
    if dest is not None:
        if not isinstance(data, Path):
            raise TypeError("Expected generated code download path when destination is provided")
        return data
    if not isinstance(data, bytes):
        raise TypeError("Expected generated code bytes when no destination path is provided")
    return data


def status(
    project_id: str,
    task_id: str,
    *,
    client: Optional[DagnamClient] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> JsonObject:
    """Check the status of a code generation task."""
    resolved = resolve_client(client, api_key, api_url)
    return resolved.get_code_status(project_id, task_id)


__all__ = [
    "download",
    "generate",
    "preview",
    "status",
    "validate",
]
