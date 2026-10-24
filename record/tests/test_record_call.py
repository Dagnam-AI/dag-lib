"""``record_call``: the escape hatch goes through the same line building and writer, and never raises."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

from _api_support import REQUEST, RESPONSE, finish_lines, record, saved_counts
import pytest

import dagnam_record
from dagnam_record import _state
from dagnam_record._files import purge

pytestmark = pytest.mark.usefixtures("fresh_state")


def test_a_recorded_call_is_a_valid_line_with_the_given_fields(
    tmp_path: Path, validate_line: Any
) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    record()
    dagnam_record.uninstall()
    (path,) = (tmp_path / "t").glob("*.jsonl")
    (text,) = path.read_text(encoding="utf-8").splitlines()
    validate_line(text)
    line = json.loads(text)
    assert (line["provider"], line["endpoint"], line["status"], line["latency_ms"]) == (
        "other",
        "custom",
        200,
        12.5,
    )
    assert (line["request"], line["response"]) == (REQUEST, RESPONSE)
    assert "norm" not in line


def test_the_defaults_and_the_other_fields(tmp_path: Path, chat_bodies: Any) -> None:
    request, response = chat_bodies
    dagnam_record.install(dir=str(tmp_path / "t"))
    dagnam_record.record_call(provider="other", request={"a": 1}, response={"b": 2})
    dagnam_record.record_call(
        provider="openai",
        endpoint="chat.completions",
        request=request,
        response=response,
        status=429,
        latency_ms=3,
    )
    custom, chat = finish_lines(tmp_path / "t")
    assert (custom["latency_ms"], custom["status"], custom["endpoint"]) == (0, 200, "custom")
    assert (chat["status"], chat["norm"]["response"], chat["latency_ms"]) == (429, "Hello", 3)


def test_sdk_style_objects_in_the_bodies_become_plain_json(tmp_path: Path) -> None:
    class Model:
        def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
            return {"dumped": True}

    dagnam_record.install(dir=str(tmp_path / "t"))
    record(
        request={"model": Model(), "when": datetime(2026, 1, 2, tzinfo=timezone.utc)},
        response={"raw": b"abc"},
    )
    (line,) = finish_lines(tmp_path / "t")
    assert line["request"] == {"model": {"dumped": True}, "when": "2026-01-02T00:00:00+00:00"}
    assert line["response"] == {"raw": {"type": "media_omitted", "bytes": 3}}


def circular() -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    body["me"] = body
    return body


MALFORMED: List[Dict[str, Any]] = [
    {"provider": "nope"},
    {"provider": 5},
    {"endpoint": "nope"},
    {"request": "text"},
    {"response": None},
    {"request": object()},
    {"request": circular()},
    {"latency_ms": "slow"},
    {"latency_ms": float("nan")},
    {"status": 42},
    {"status": "200"},
    {"request": {"n": float("nan")}},
    {"request": {"s": "\ud800"}},
]


@pytest.mark.parametrize("overrides", MALFORMED, ids=[str(i) for i in range(len(MALFORMED))])
def test_a_malformed_call_is_swallowed_and_counted(
    overrides: Dict[str, Any], tmp_path: Path
) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    record(**overrides)
    record()  # recording goes on
    assert len(finish_lines(tmp_path / "t")) == 1
    counts = saved_counts(tmp_path / "t")
    assert (counts["dropped_error"], counts["written"]) == (1, 1)


def test_an_oversize_call_is_counted_as_too_large(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    record(request={"prompt": "x" * 1_100_000})
    assert finish_lines(tmp_path / "t") == []
    counts = saved_counts(tmp_path / "t")
    assert (counts["dropped_large"], counts["dropped_error"]) == (1, 0)


def test_a_body_with_too_many_parts_is_counted_as_too_large(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    shared = list(range(1000))
    record(request={"parts": [shared] * 300})
    finish_lines(tmp_path / "t")
    assert saved_counts(tmp_path / "t")["dropped_large"] == 1


def test_a_writer_that_raises_is_counted_and_never_reaches_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    installed = _state.STATE.installed
    assert installed is not None

    def broken(line: str) -> None:
        raise RuntimeError("writer broke")

    monkeypatch.setattr(installed.writer, "submit", broken)
    record()
    monkeypatch.undo()
    assert saved_counts_after(tmp_path / "t")["dropped_error"] == 1


def saved_counts_after(directory: Path) -> Dict[str, int]:
    dagnam_record.uninstall()
    return saved_counts(directory)


def test_before_install_it_records_nothing_and_warns_once(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    with caplog.at_level(logging.WARNING, logger="dagnam_record"):
        record()
        record()
    assert caplog.text.count("record_call() ran without install()") == 1
    assert not (tmp_path / "t").exists()


def test_after_uninstall_it_records_nothing(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    record()
    dagnam_record.uninstall()
    record()
    assert len(finish_lines(tmp_path / "t")) == 1


def test_under_the_kill_switch_it_is_silent(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    dagnam_record.install(dir=str(tmp_path / "t"))
    with caplog.at_level(logging.WARNING, logger="dagnam_record"):
        record()
    assert caplog.text == ""
    assert not (tmp_path / "t").exists()


def wait_for_lines(directory: Path, wanted: int) -> List[str]:
    """The lines on disk once there are ``wanted`` of them (the writer thread is asynchronous)."""
    deadline = time.monotonic() + 10
    found: List[str] = []
    while time.monotonic() < deadline:
        found = [
            line
            for path in sorted(directory.glob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if len(found) >= wanted:
            break
        time.sleep(0.01)
    return found


def test_after_a_purge_the_running_recorder_starts_a_new_file(tmp_path: Path) -> None:
    directory = tmp_path / "t"
    dagnam_record.install(dir=str(directory))
    record(request={"prompt": "before"})
    assert len(wait_for_lines(directory, 1)) == 1
    name = next(directory.glob("*.jsonl")).name
    assert purge(str(directory)).deleted == [name]
    record(request={"prompt": "after"})
    (line,) = wait_for_lines(directory, 1)
    assert json.loads(line)["request"]["prompt"] == "after"
