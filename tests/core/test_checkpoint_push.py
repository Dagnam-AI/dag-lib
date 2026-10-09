"""``push_checkpoint``: the best-effort mid-run checkpoint push, against the documented route.

``POST /api/v1/training/jobs/{id}/checkpoints/push?epoch=&step=&name=`` with the raw file (or an
uncompressed tar of a directory) as the body, authorized by the run token in the environment.
Every refusal the route documents is logged and skipped; nothing may stop the training loop.
"""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
from typing import TYPE_CHECKING, Any

import pytest
import requests
import requests_mock as rm_module

from dagnam._core import checkpoint_push
from dagnam._core.checkpoint_push import push_checkpoint

if TYPE_CHECKING:
    from tests.typing_helpers import PytestMonkeyPatch, RequestsMocker

API = "https://api.test"
PUSH = f"{API}/api/v1/training/jobs/job-1/checkpoints/push"
RUN_TOKEN = "run-token-SECRET-123"


class _Log:
    """Collects what the training script would write to its log."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def __call__(self, level: str, message: str) -> None:
        self.lines.append((level, message))

    @property
    def text(self) -> str:
        return "\n".join(message for _, message in self.lines)


@pytest.fixture(autouse=True)
def platform_run(monkeypatch: PytestMonkeyPatch) -> None:
    """A training script on the platform: job id, run token and API url in the environment."""
    monkeypatch.delenv("DAGNAM_INTERNAL", raising=False)
    monkeypatch.setenv("DAGNAM_JOB_ID", "job-1")
    monkeypatch.setenv("DAGNAM_API_KEY", RUN_TOKEN)
    monkeypatch.setenv("DAGNAM_API_URL", API)
    monkeypatch.setattr(checkpoint_push, "_stopped", False)


@pytest.fixture
def rmock():
    with rm_module.Mocker() as m:
        yield m


@pytest.fixture
def log() -> _Log:
    return _Log()


def _saved(rmock: RequestsMocker, **body: object) -> None:
    rmock.post(
        PUSH,
        status_code=201,
        json={"checkpoint_id": "ck-1", "epoch": 3, "step": 900, "size_bytes": 4, **body},
    )


def _file(tmp_path: Path, name: str = "checkpoint_epoch_3.pth", data: bytes = b"WEIGHTS") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ------------------------------------------------------------------ what is sent


def test_a_file_is_sent_raw_with_the_run_token(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    received: list[bytes] = []

    def reply(request: Any, context: Any) -> dict[str, object]:
        received.append(request.body.read())
        context.status_code = 201
        return {"checkpoint_id": "ck-1"}

    rmock.post(PUSH, json=reply)

    saved = push_checkpoint(_file(tmp_path), epoch=3, step=900, log=log)

    assert saved == "ck-1"
    assert received == [b"WEIGHTS"]
    sent = rmock.last_request
    assert sent.headers["Authorization"] == f"Bearer {RUN_TOKEN}"
    assert sent.headers["Content-Type"] == "application/octet-stream"
    assert sent.headers["Content-Length"] == "7"
    assert "Transfer-Encoding" not in sent.headers
    assert sent.qs == {"epoch": ["3"], "step": ["900"], "name": ["checkpoint_epoch_3.pth"]}
    assert sent.timeout == (10, 900)
    assert log.lines == []


def test_a_directory_is_sent_as_one_uncompressed_tar_with_its_exact_length(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    root = tmp_path / "ckpt-dir"
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"A" * 600)
    (root / "sub" / "b.json").write_text("{}")
    received: list[bytes] = []

    def reply(request: Any, context: Any) -> dict[str, object]:
        received.append(b"".join(request.body))
        context.status_code = 201
        return {"checkpoint_id": "ck-2"}

    rmock.post(PUSH, json=reply)

    assert push_checkpoint(root, epoch=1, step=10, log=log) == "ck-2"

    sent = rmock.last_request
    assert sent.qs["name"] == ["ckpt-dir.tar"]
    assert sent.headers["Content-Length"] == str(len(received[0]))
    assert "Transfer-Encoding" not in sent.headers
    with tarfile.open(fileobj=io.BytesIO(received[0])) as tar:
        assert sorted(tar.getnames()) == ["a.bin", "sub/b.json"]


def test_the_path_may_be_a_pathlike_or_a_string(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    _saved(rmock)
    assert push_checkpoint(str(_file(tmp_path)), epoch=0, step=0, log=log) == "ck-1"
    assert push_checkpoint(_file(tmp_path), epoch=0, step=0, log=log) == "ck-1"


def test_a_relative_path_is_named_by_what_it_resolves_to(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    _file(tmp_path, "latest.pt")
    monkeypatch.chdir(tmp_path)
    _saved(rmock)
    assert push_checkpoint("./sub/../latest.pt", epoch=0, step=0, log=log) == "ck-1"
    assert rmock.last_request.qs["name"] == ["latest.pt"]


def test_a_response_without_an_id_saves_nothing_to_report(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    rmock.post(PUSH, status_code=201, json={"epoch": 3})
    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert log.lines == []


@pytest.mark.parametrize(
    ("name", "directory", "expected"),
    [
        ("checkpoint_epoch_3.pth", False, "checkpoint_epoch_3.pth"),
        ("my ckpt (1).pth", False, "my_ckpt__1_.pth"),
        ("naïve.weights.h5", False, "na_ve.weights.h5"),
        ("step-500", True, "step-500.tar"),
        ("x" * 200 + ".pth", False, "x" * 116 + ".pth"),
        ("y" * 200, True, "y" * 116 + ".tar"),
        ("/", True, "checkpoint.tar"),
    ],
)
def test_the_stored_name_is_one_the_platform_accepts(
    name: str, directory: bool, expected: str
) -> None:
    assert checkpoint_push.upload_name(Path("/ck", name), directory=directory) == expected


# ----------------------------------------------------- the route's refusals, skipped


@pytest.mark.parametrize(
    ("status", "body", "shown"),
    [
        (409, {"detail": {"error": "push_in_progress", "message": "busy"}}, "HTTP 409"),
        (409, {"detail": "another push is running"}, "HTTP 409"),
        (411, {"detail": "length required"}, "HTTP 411"),
        (413, {"detail": "checkpoint too large"}, "HTTP 413"),
        (429, {"detail": "slow down"}, "HTTP 429"),
        (500, {"detail": "boom"}, "HTTP 500"),
        (503, {"detail": "storage down"}, "HTTP 503"),
        (401, {"detail": "expired"}, "AuthError"),
        (404, {"detail": "no such job"}, "TrainingJobNotFoundError"),
        (422, {"detail": "bad epoch"}, "HTTP 422"),
    ],
)
def test_a_refusal_is_logged_and_skipped(
    tmp_path: Path,
    rmock: RequestsMocker,
    log: _Log,
    status: int,
    body: dict[str, object],
    shown: str,
) -> None:
    rmock.post(PUSH, status_code=status, json=body)

    assert push_checkpoint(_file(tmp_path), epoch=3, step=900, log=log) is None

    assert [level for level, _ in log.lines] == ["WARNING"]
    assert shown in log.text
    assert "epoch 3, step 900" in log.text
    assert "training continues" in log.text


@pytest.mark.parametrize("exc", [requests.ConnectionError("down"), requests.Timeout("slow")])
def test_a_network_failure_is_logged_and_skipped(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, exc: Exception
) -> None:
    rmock.post(PUSH, exc=exc)
    assert push_checkpoint(_file(tmp_path), epoch=3, step=900, log=log) is None
    assert "network error" in log.text


def test_nothing_in_the_log_carries_a_secret_a_url_or_a_server_body(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    rmock.post(PUSH, status_code=500, text=f"internal: {RUN_TOKEN} at {PUSH}")
    push_checkpoint(_file(tmp_path), epoch=3, step=900, log=log)
    rmock.post(PUSH, exc=requests.ConnectionError(f"HTTPSConnectionPool: {PUSH}?token=abc"))
    push_checkpoint(_file(tmp_path), epoch=4, step=901, log=log)

    assert len(log.lines) == 2
    for secret in (RUN_TOKEN, "api.test", "http", "token="):
        assert secret not in log.text


def test_a_path_that_is_not_there_is_logged_without_calling_the_platform(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    assert push_checkpoint(tmp_path / "missing.pth", epoch=3, step=1, log=log) is None
    assert rmock.call_count == 0
    assert "FileNotFoundError" in log.text


def test_a_run_without_credentials_is_logged_not_raised(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.delenv("DAGNAM_API_KEY")
    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert rmock.call_count == 0
    assert "AuthError" in log.text


def test_ctrl_c_is_not_swallowed(tmp_path: Path, rmock: RequestsMocker, log: _Log) -> None:
    rmock.post(PUSH, exc=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log)


# --------------------------------------------- a job that no longer takes checkpoints


def test_a_job_that_stopped_taking_checkpoints_ends_the_pushing_quietly(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    rmock.post(
        PUSH,
        status_code=409,
        json={"detail": {"error": "not_accepting_checkpoints", "message": "not running"}},
    )

    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert push_checkpoint(_file(tmp_path), epoch=4, step=2, log=log) is None

    assert rmock.call_count == 1
    assert log.lines == []


def test_a_busy_409_does_not_end_the_pushing(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    rmock.post(
        PUSH,
        [
            {"status_code": 409, "json": {"detail": "busy"}},
            {"status_code": 201, "json": {"checkpoint_id": "ck-9"}},
        ],
    )

    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert push_checkpoint(_file(tmp_path), epoch=4, step=2, log=log) == "ck-9"
    assert rmock.call_count == 2


# --------------------------------------------------------------- not on the platform


def test_the_platforms_own_host_collects_checkpoints_itself(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_INTERNAL", "1")
    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert rmock.call_count == 0
    assert log.lines == []


def test_a_local_run_has_nowhere_to_push(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.delenv("DAGNAM_JOB_ID")
    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert rmock.call_count == 0
    assert log.lines == []
