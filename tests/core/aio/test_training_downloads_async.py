"""Wire-level coverage for the async training-download client methods.

Async mirror of ``tests/core/client/test_training_downloads.py``: streaming
download of a job's generated code ZIP and DAG JSON, including the traversal-safe
filename handling and the streaming connect/timeout error wrapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from dagnam._core.aio import AsyncDagnamClient
from dagnam._core.exceptions import APIError, TrainingJobNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import RespxMockRouter

pytestmark = pytest.mark.anyio

CODE = "/api/v1/training/jobs/j1/download-code"
DAG = "/api/v1/training/jobs/j1/dag"


# ---------------------------------------------------------------- download-code


async def test_download_code_writes_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(CODE).mock(
        return_value=httpx.Response(
            200,
            content=b"zip-bytes",
            headers={"content-disposition": 'attachment; filename="proj-pytorch.zip"'},
        )
    )
    out = await client.download_training_code("j1", tmp_path)
    assert out.name == "proj-pytorch.zip"
    assert out.read_bytes() == b"zip-bytes"


async def test_download_code_default_name_when_header_absent(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(CODE).mock(return_value=httpx.Response(200, content=b"zip-bytes"))
    out = await client.download_training_code("j1", tmp_path)
    assert out.name == "j1-code.zip"


async def test_download_code_404_raises_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(CODE).mock(return_value=httpx.Response(404, json={"detail": "no code"}))
    with pytest.raises(TrainingJobNotFoundError):
        await client.download_training_code("j1", tmp_path)


async def test_download_code_connect_error(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(CODE).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(APIError, match="Connection failed"):
        await client.download_training_code("j1", tmp_path)


async def test_download_code_timeout(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(CODE).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(APIError, match="Request timed out"):
        await client.download_training_code("j1", tmp_path)


async def test_download_code_traversal_filename_lands_inside_dest_dir(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    """Async mirror of the mandatory traversal test."""
    mock.get(CODE).mock(
        return_value=httpx.Response(
            200,
            content=b"zip-bytes",
            headers={"content-disposition": 'attachment; filename="../../etc/passwd"'},
        )
    )
    out = await client.download_training_code("j1", tmp_path)
    assert out.parent == tmp_path
    assert out == tmp_path / "passwd"
    assert out.read_bytes() == b"zip-bytes"


# ------------------------------------------------------------------- download-dag


async def test_download_dag_writes_file(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(DAG).mock(
        return_value=httpx.Response(
            200,
            content=b'{"nodes": []}',
            headers={"content-disposition": 'attachment; filename="proj-dag.json"'},
        )
    )
    out = await client.download_dag("j1", tmp_path)
    assert out.name == "proj-dag.json"
    assert out.read_bytes() == b'{"nodes": []}'


async def test_download_dag_default_name_when_header_absent(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(DAG).mock(return_value=httpx.Response(200, content=b"{}"))
    out = await client.download_dag("j1", tmp_path)
    assert out.name == "j1-dag.json"


async def test_download_dag_404_raises_not_found(
    client: AsyncDagnamClient, mock: RespxMockRouter, tmp_path: Path
) -> None:
    mock.get(DAG).mock(return_value=httpx.Response(404, json={"detail": "no dag"}))
    with pytest.raises(TrainingJobNotFoundError):
        await client.download_dag("j1", tmp_path)
