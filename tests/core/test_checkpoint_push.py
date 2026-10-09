"""``push_checkpoint``: the best-effort mid-run checkpoint push, against the documented route.

``POST /api/v1/training/jobs/{id}/checkpoints/push?epoch=&step=&name=`` with the raw file (or an
uncompressed tar of a directory) as the body, authorized by the run token in the environment.
Every refusal the route documents is logged and skipped; nothing may stop the training loop.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import tarfile
from typing import TYPE_CHECKING, Any

import pytest
import requests
import requests_mock as rm_module
from tests.core.client.test_common_credits import TRAINING_BODY

from dagnam._core import auth, checkpoint_push
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
    monkeypatch.setattr(checkpoint_push, "_not_before", 0.0)


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
        received.append(b"".join(request.body))
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
        (408, {"detail": "slow client"}, "HTTP 408"),
        (500, {"detail": "boom"}, "HTTP 500"),
        (503, {"detail": "storage down"}, "HTTP 503"),
    ],
)
def test_a_passing_refusal_is_logged_and_the_next_push_is_tried(
    tmp_path: Path,
    rmock: RequestsMocker,
    log: _Log,
    status: int,
    body: dict[str, object],
    shown: str,
) -> None:
    rmock.post(
        PUSH,
        [
            {"status_code": status, "json": body},
            {"status_code": 201, "json": {"checkpoint_id": "ck-2"}},
        ],
    )

    assert push_checkpoint(_file(tmp_path), epoch=3, step=900, log=log) is None

    assert [level for level, _ in log.lines] == ["WARNING"]
    assert shown in log.text
    assert "epoch 3, step 900" in log.text
    assert "training continues" in log.text
    assert push_checkpoint(_file(tmp_path), epoch=4, step=901, log=log) == "ck-2"


@pytest.mark.parametrize(
    ("status", "shown"),
    [
        (400, "HTTP 400"),
        (401, "HTTP 401"),
        (402, "HTTP 402"),
        (403, "HTTP 403"),
        (404, "HTTP 404"),
        (405, "HTTP 405"),
        (410, "HTTP 410"),
        (411, "HTTP 411"),
        (413, "HTTP 413"),
        (422, "HTTP 422"),
    ],
)
def test_a_refusal_that_will_repeat_is_warned_once_and_ends_the_pushing(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, status: int, shown: str
) -> None:
    rmock.post(PUSH, status_code=status, json={"detail": "no"})

    for epoch in (3, 4, 5):
        assert push_checkpoint(_file(tmp_path), epoch=epoch, step=epoch, log=log) is None

    assert rmock.call_count == 1
    assert [level for level, _ in log.lines] == ["WARNING"]
    assert shown in log.text
    assert "no more checkpoints will be pushed" in log.text


def test_a_platform_without_the_route_produces_one_warning_per_run(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    rmock.post(PUSH, status_code=404, json={"detail": "Not Found"})
    for epoch in range(10):
        push_checkpoint(_file(tmp_path), epoch=epoch, step=epoch, log=log)
    assert rmock.call_count == 1
    assert len(log.lines) == 1


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: PytestMonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(checkpoint_push.time, "monotonic", fake)
    return fake


@pytest.mark.parametrize(
    ("header", "wait"),
    [("120", 120.0), (None, 300.0), ("soon", 300.0), ("0", 1.0), ("999999", 3600.0)],
)
def test_a_rate_limit_is_silent_and_waits_as_long_as_the_platform_says(
    tmp_path: Path,
    rmock: RequestsMocker,
    log: _Log,
    clock: _Clock,
    header: str | None,
    wait: float,
) -> None:
    headers = {} if header is None else {"Retry-After": header}
    rmock.post(
        PUSH,
        [
            {"status_code": 429, "json": {"detail": "slow"}, "headers": headers},
            {"status_code": 201, "json": {"checkpoint_id": "ck-3"}},
        ],
    )

    assert push_checkpoint(_file(tmp_path), epoch=1, step=1, log=log) is None
    clock.now += wait - 1
    assert push_checkpoint(_file(tmp_path), epoch=2, step=2, log=log) is None
    assert rmock.call_count == 1
    clock.now += 1
    assert push_checkpoint(_file(tmp_path), epoch=3, step=3, log=log) == "ck-3"
    assert rmock.call_count == 2
    assert log.lines == []


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


def test_a_run_without_the_run_token_is_not_a_platform_run(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.delenv("DAGNAM_API_KEY")
    monkeypatch.setattr(auth, "_api_key", "user-key")
    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) is None
    assert rmock.call_count == 0
    assert log.lines == []


def test_the_run_token_and_url_come_only_from_the_environment(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "_api_key", "user-key")
    monkeypatch.setattr(auth, "_api_url", "https://elsewhere.test")
    _saved(rmock)

    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) == "ck-1"

    assert rmock.last_request.url.startswith(API)
    assert rmock.last_request.headers["Authorization"] == f"Bearer {RUN_TOKEN}"


def test_without_a_url_in_the_environment_the_default_platform_is_used(
    tmp_path: Path, rmock: RequestsMocker, log: _Log, monkeypatch: PytestMonkeyPatch
) -> None:
    monkeypatch.delenv("DAGNAM_API_URL")
    monkeypatch.setattr(auth, "_api_url", "https://elsewhere.test")
    rmock.post(
        "https://api.dagnam.ai/api/v1/training/jobs/job-1/checkpoints/push",
        status_code=201,
        json={"checkpoint_id": "ck-d"},
    )
    assert push_checkpoint(_file(tmp_path), epoch=3, step=1, log=log) == "ck-d"


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


# ------------------------------------------------- what is not a checkpoint to send


def test_a_pipe_is_refused_without_blocking(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    assert push_checkpoint(pipe, epoch=1, step=1, log=log) is None
    assert rmock.call_count == 0
    assert "OSError" in log.text


def test_a_device_is_refused(tmp_path: Path, rmock: RequestsMocker, log: _Log) -> None:
    assert push_checkpoint("/dev/zero", epoch=1, step=1, log=log) is None
    assert rmock.call_count == 0
    assert "OSError" in log.text


def test_an_empty_file_is_skipped_not_sent_chunked(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    assert push_checkpoint(_file(tmp_path, data=b""), epoch=1, step=1, log=log) is None
    assert rmock.call_count == 0
    assert log.lines == []


def test_a_directory_with_no_files_is_skipped(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    empty = tmp_path / "ckpt"
    empty.mkdir()
    (empty / "link").symlink_to(tmp_path)
    assert push_checkpoint(empty, epoch=1, step=1, log=log) is None
    assert rmock.call_count == 0
    assert log.lines == []


def test_a_symlinked_checkpoint_path_is_the_users_choice_and_is_followed(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    real = _file(tmp_path, "checkpoint_epoch_9.pth")
    (tmp_path / "latest.pth").symlink_to(real)
    _saved(rmock)
    assert push_checkpoint(tmp_path / "latest.pth", epoch=9, step=1, log=log) == "ck-1"
    assert rmock.last_request.qs["name"] == ["latest.pth"]


def test_a_file_that_changes_while_it_is_sent_is_a_logged_failure_not_a_checkpoint(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    victim = _file(tmp_path, data=b"S" * 3000)
    seen: list[bytes] = []

    def reply(request: Any, context: Any) -> dict[str, object]:
        victim.write_bytes(b"S" * 10)
        try:
            seen.append(b"".join(request.body))
        except OSError as exc:
            # What urllib3 does with an error raised by a body: a connection error.
            raise requests.ConnectionError("Connection aborted.") from exc
        context.status_code = 201
        return {"checkpoint_id": "ck-x"}

    rmock.post(PUSH, json=reply)

    assert push_checkpoint(victim, epoch=1, step=1, log=log) is None

    assert seen == []
    assert "changed while it was being sent" in log.text
    assert "network error" not in log.text


def test_a_file_that_vanishes_while_it_is_sent_says_so_and_not_network_error(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    victim = _file(tmp_path)

    def reply(request: Any, context: Any) -> dict[str, object]:
        victim.unlink()
        try:
            b"".join(request.body)
        except OSError as exc:
            raise requests.ConnectionError("Connection aborted.") from exc
        return {}

    rmock.post(PUSH, json=reply)

    assert push_checkpoint(victim, epoch=1, step=1, log=log) is None

    assert "FileNotFoundError" in log.text
    assert "network error" not in log.text


def test_an_unfunded_account_is_warned_once_and_ends_the_pushing(
    tmp_path: Path, rmock: RequestsMocker, log: _Log
) -> None:
    # The typed credit refusal is a QuotaExceededError too: it is not an APIError.
    rmock.post(PUSH, status_code=402, json=TRAINING_BODY)

    for epoch in (3, 4, 5):
        assert push_checkpoint(_file(tmp_path), epoch=epoch, step=epoch, log=log) is None

    assert rmock.call_count == 1
    assert len(log.lines) == 1
    assert "HTTP 402" in log.text
    assert "no more checkpoints will be pushed" in log.text
