"""Wire-level coverage for the sync training-download client methods.

Covers ``TrainingClientMixin.download_training_code`` and ``download_dag``,
including the mandatory traversal test proving a hostile ``Content-Disposition``
filename lands inside ``dest_dir`` rather than escaping it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import TrainingJobNotFoundError

if TYPE_CHECKING:
    from tests.typing_helpers import RequestsMocker

API = "https://api.test"
CODE = f"{API}/api/v1/training/jobs/j1/download-code"
DAG = f"{API}/api/v1/training/jobs/j1/dag"


# ---------------------------------------------------------------- download-code


def test_download_code_writes_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(
        CODE,
        content=b"zip-bytes",
        headers={"Content-Disposition": 'attachment; filename="proj-pytorch.zip"'},
    )
    out = client.download_training_code("j1", tmp_path)
    assert out.name == "proj-pytorch.zip"
    assert out.parent == tmp_path
    assert out.read_bytes() == b"zip-bytes"


def test_download_code_default_name_when_header_absent(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(CODE, content=b"zip-bytes")
    out = client.download_training_code("j1", tmp_path)
    assert out.name == "j1-code.zip"


def test_download_code_404_raises_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(CODE, status_code=404, text="no generated code")
    with pytest.raises(TrainingJobNotFoundError):
        client.download_training_code("j1", tmp_path)


def test_download_code_traversal_filename_lands_inside_dest_dir(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    """A hostile Content-Disposition filename is reduced to its basename."""
    rmock.get(
        CODE,
        content=b"zip-bytes",
        headers={"Content-Disposition": 'attachment; filename="../../etc/passwd"'},
    )
    out = client.download_training_code("j1", tmp_path)
    assert out.parent == tmp_path
    assert out == tmp_path / "passwd"
    assert out.read_bytes() == b"zip-bytes"
    assert not Path("/etc/passwd").exists() or Path("/etc/passwd").read_bytes() != b"zip-bytes"


# ------------------------------------------------------------------- download-dag


def test_download_dag_writes_file(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(
        DAG,
        content=b'{"nodes": []}',
        headers={"Content-Disposition": 'attachment; filename="proj-dag.json"'},
    )
    out = client.download_dag("j1", tmp_path)
    assert out.name == "proj-dag.json"
    assert out.read_bytes() == b'{"nodes": []}'


def test_download_dag_default_name_when_header_absent(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(DAG, content=b"{}")
    out = client.download_dag("j1", tmp_path)
    assert out.name == "j1-dag.json"


def test_download_dag_404_raises_not_found(
    client: DagnamClient, rmock: RequestsMocker, tmp_path: Path
) -> None:
    rmock.get(DAG, status_code=404, text="no dag")
    with pytest.raises(TrainingJobNotFoundError):
        client.download_dag("j1", tmp_path)
