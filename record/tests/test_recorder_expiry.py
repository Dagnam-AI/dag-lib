"""After the window ends, or once the kill switch is flipped, a call costs the recorder nothing."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List

from _api_support import envelope, finish_lines, record, saved_counts, write_data
from _mock_server import Reply
from _openai_bodies import CHAT_PATH, MODEL, chat_body
from _openai_calls import unpatched
import httpx
import openai
from openai.resources.chat.completions import Completions
import pytest

import dagnam_record
from dagnam_record import _openai_body, _recorder

DAY = 86400
MESSAGES: List[Any] = [{"role": "user", "content": "Hi"}]


@pytest.fixture
def work(monkeypatch: pytest.MonkeyPatch) -> Dict[str, int]:
    """Counts how often a line is built and a request converted; the real functions still run."""
    calls = {"build_line": 0, "to_plain": 0}
    real_build, real_plain = _recorder.build_line, _openai_body.to_plain

    def build(**arguments: Any) -> str:
        calls["build_line"] += 1
        return real_build(**arguments)

    def plain(value: Any) -> Any:
        calls["to_plain"] += 1
        return real_plain(value)

    monkeypatch.setattr(_recorder, "build_line", build)
    monkeypatch.setattr(_openai_body, "to_plain", plain)
    return calls


def seed_status(directory: Path, started: float) -> None:
    """A status file of an earlier run that began at ``started``."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    (directory / ".status.json").write_text(
        json.dumps({"started": started, "counts": {"written": 3}}), encoding="utf-8"
    )


@pytest.mark.usefixtures("fresh_state")
def test_the_kill_switch_is_read_on_every_call_not_only_by_install(
    tmp_path: Path, mock_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "traces"
    assert dagnam_record.install(dir=str(directory))
    mock_server.on(CHAT_PATH, Reply(body=chat_body()))
    client = _client(mock_server)
    client.chat.completions.create(model=MODEL, messages=MESSAGES)
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    result = client.chat.completions.create(model=MODEL, messages=MESSAGES)
    record()
    assert result.choices[0].message.content == "Hello there"
    monkeypatch.delenv("DAGNAM_RECORD")
    client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert len(finish_lines(directory)) == 2  # the first and the last, not the switched-off one


def _client(server: Any) -> Any:
    return openai.OpenAI(
        api_key="k",
        base_url=server.url,
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
    )


@pytest.mark.usefixtures("fresh_state")
def test_install_after_the_window_ended_patches_nothing_and_says_so_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, work: Dict[str, int]
) -> None:
    directory = tmp_path / "traces"
    seed_status(directory, time.time() - 30 * DAY)
    with caplog.at_level(logging.INFO, logger="dagnam_record"):
        assert dagnam_record.install(dir=str(directory)) is False
        assert dagnam_record.install(dir=str(directory)) is False
    assert not hasattr(Completions.create, "__dagnam_record__")
    ended = [r for r in caplog.records if "window has ended" in r.getMessage()]
    assert len(ended) == 1
    record()
    record()
    assert work == {"build_line": 0, "to_plain": 0}
    assert finish_lines(directory) == []
    assert saved_counts(directory)["dropped_expired"] == 2
    assert saved_counts(directory)["written"] == 3


def test_a_window_that_ends_in_a_running_process_stops_the_work_at_once(
    rig: Any, work: Dict[str, int]
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
        first = dict(work)
        rig.writer.drain()
        assert first["build_line"] == 1
        rig.writer.clock().now += 15 * DAY  # the 14 days are over; no write has seen it yet
        result = client.chat.completions.create(model=MODEL, messages=MESSAGES)
        plain = unpatched(Completions.create)(
            client.chat.completions, model=MODEL, messages=MESSAGES
        )
    assert result == plain
    assert work == first  # no conversion, no line for the second call
    assert rig.writer.expired
    assert rig.writer.counts["dropped_expired"] == 1
    assert len(rig.lines()) == 1


def test_a_call_that_began_before_the_end_and_finished_after_it_is_counted_not_built(
    rig: Any, work: Dict[str, int]
) -> None:
    rig.writer.clock().now += 15 * DAY
    rig.recorder.record("other", "custom", {"a": 1}, {"b": 2}, status=200, latency_ms=1.0)
    assert work["build_line"] == 0
    assert rig.writer.counts["dropped_expired"] == 1


@pytest.mark.usefixtures("fresh_state")
def test_files_older_than_the_window_are_kept_once_the_window_has_ended(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    seed_status(directory, time.time() - 30 * DAY)
    old = write_data(directory, "2020-01-01.jsonl", [envelope("old", 1_577_880_000.0)])
    dagnam_record.install(dir=str(directory))
    assert old.exists()  # a restart after the window must not eat the file waiting to be uploaded
    dagnam_record.uninstall()


@pytest.mark.usefixtures("fresh_state")
def test_files_older_than_the_window_are_still_pruned_inside_it(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    seed_status(directory, time.time() - 2 * DAY)
    old = write_data(directory, "2020-01-01.jsonl", [envelope("old", 1_577_880_000.0)])
    dagnam_record.install(dir=str(directory), window_days=7)
    assert not old.exists()
    dagnam_record.uninstall()
    assert os.path.exists(directory / ".status.json")
