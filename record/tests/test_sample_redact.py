"""Sampling by a hash of the trace id, the ``redact`` hook (callable and ``"auto"``) and ``no_raw``."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List
import uuid

from _api_support import finish_lines, record, saved_counts
import pytest

import dagnam_record
from dagnam_record import _recorder
from dagnam_record._recorder import Recorder, sampled

pytestmark = pytest.mark.usefixtures("fresh_state")
EMAIL = "someone.private@example.org"


def kept(rate: float, ids: int = 2000) -> int:
    return sum(sampled(f"trace-{number}", rate) for number in range(ids))


def test_a_half_sample_keeps_between_40_and_60_percent_of_2000_ids() -> None:
    assert 800 <= kept(0.5) <= 1200
    assert 300 <= kept(0.25) <= 700


@pytest.mark.parametrize(
    ("trace_id", "fraction"),
    [
        ("trace-0", 0.6086722470968304),
        ("trace-1", 0.09372610975976327),
        ("trace-2", 0.853069755933122),
        ("abc", 0.7283949105904022),
        ("00000000-0000-0000-0000-000000000001", 0.4795184636032407),
    ],
)
def test_the_decision_for_a_fixed_id_is_fixed_for_ever(trace_id: str, fraction: float) -> None:
    # sha256 of the id, first 8 bytes: the same on every run, every machine, every Python.
    assert sampled(trace_id, fraction) is False  # the share is "below the rate", not "at most"
    assert sampled(trace_id, fraction + 1e-9) is True
    assert sampled(trace_id, fraction - 1e-9) is False
    assert sampled(trace_id, 1.0) is True


def test_the_decision_depends_only_on_the_id_and_larger_rates_keep_more() -> None:
    ids = [f"id-{number}" for number in range(500)]
    assert [sampled(item, 0.3) for item in ids] == [sampled(item, 0.3) for item in ids]
    for item in ids:
        if sampled(item, 0.3):
            assert sampled(item, 0.6)
            assert sampled(item, 1.0)
    assert all(sampled(item, 1.0) for item in ids)
    assert not any(sampled(item, 1e-12) for item in ids)


def test_a_recorder_records_the_calls_the_hash_selects(make: Any, tmp_path: Path) -> None:
    writer = make(max_queue=5000)
    recorder = Recorder(writer, sample=0.5)
    for _ in range(2000):
        recorder.record("other", "custom", {"p": 1}, {"r": 2}, status=200, latency_ms=1.0)
    writer.drain()
    lines = [
        json.loads(line)
        for path in (tmp_path / "traces").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert 800 <= len(lines) <= 1200
    assert all(sampled(line["trace_id"], 0.5) for line in lines)
    assert writer.counts["written"] == len(lines)


def test_a_retried_call_with_the_same_trace_id_is_kept_or_dropped_as_one(
    make: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [uuid.UUID(int=number) for number in range(200)]
    seen = iter(ids * 2)
    monkeypatch.setattr(_recorder.uuid, "uuid4", lambda: next(seen))
    writer = make()
    recorder = Recorder(writer, sample=0.5)
    for _ in range(400):
        recorder.record("other", "custom", {"p": 1}, {"r": 2}, status=200, latency_ms=1.0)
    writer.drain()
    stored = [
        json.loads(line)["trace_id"]
        for path in (tmp_path / "traces").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    expected = [str(item) for item in ids if sampled(str(item), 0.5)]
    assert stored == expected * 2


def test_install_with_a_sample_records_only_that_share(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"), sample=0.5)
    for _ in range(200):
        record()
    lines = finish_lines(tmp_path / "t")
    assert 50 <= len(lines) <= 150


def texts_seen() -> tuple[List[str], Callable[[str], str]]:
    seen: List[str] = []

    def mask(text: str) -> str:
        seen.append(text)
        return text.replace("hunter2", "[X]")

    return seen, mask


def test_redact_sees_the_serialized_request_and_response_and_its_result_is_recorded(
    tmp_path: Path,
) -> None:
    seen, mask = texts_seen()
    dagnam_record.install(dir=str(tmp_path / "t"), redact=mask)
    record(request={"prompt": "pw is hunter2 é"}, response={"text": "hunter2 again"})
    (line,) = finish_lines(tmp_path / "t")
    assert [json.loads(text) for text in seen] == [
        {"prompt": "pw is hunter2 é"},
        {"text": "hunter2 again"},
    ]
    assert "é" in seen[0]  # not \u-escaped, so a pattern on real characters can match
    assert line["request"] == {"prompt": "pw is [X] é"}
    assert line["response"] == {"text": "[X] again"}
    assert "hunter2" not in json.dumps(line)


def test_redact_runs_before_norm_is_built(tmp_path: Path, chat_bodies: Any) -> None:
    request, response = chat_bodies
    request = json.loads(json.dumps(request).replace("Hi", "hunter2"))
    _, mask = texts_seen()
    dagnam_record.install(dir=str(tmp_path / "t"), redact=mask)
    record(provider="openai", endpoint="chat.completions", request=request, response=response)
    (line,) = finish_lines(tmp_path / "t")
    assert "[X]" in json.dumps(line["norm"])
    assert "hunter2" not in json.dumps(line)


def bad_redactors() -> Dict[str, Callable[[str], Any]]:
    def boom(text: str) -> str:
        raise RuntimeError("redactor bug")

    return {
        "raises": boom,
        "not text": lambda text: 5,
        "bytes": lambda text: text.encode(),
        "not json": lambda text: "oops",
        "a list": lambda text: "[1]",
        "a string": lambda text: '"x"',
        "empty": lambda text: "",
    }


@pytest.mark.parametrize("name", list(bad_redactors()))
def test_a_redactor_that_fails_drops_the_call_and_never_writes_unredacted_text(
    name: str, tmp_path: Path
) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"), redact=bad_redactors()[name])
    record(request={"prompt": "hunter2"})
    assert finish_lines(tmp_path / "t") == []
    counts = saved_counts(tmp_path / "t")
    assert (counts["dropped_error"], counts["written"]) == (1, 0)
    assert "hunter2" not in "".join(p.read_text() for p in (tmp_path / "t").glob("*.jsonl"))


def test_a_redactor_is_given_nothing_but_an_object(tmp_path: Path) -> None:
    seen, mask = texts_seen()
    dagnam_record.install(dir=str(tmp_path / "t"), redact=mask)
    record(request=["not", "an", "object"])
    assert seen == []
    assert finish_lines(tmp_path / "t") == []
    assert saved_counts(tmp_path / "t")["dropped_error"] == 1


def test_redact_auto_uses_the_pii_redactor_of_dagnam_contracts(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"), redact="auto")
    record(request={"prompt": f"mail {EMAIL} please"}, response={"text": f"to {EMAIL}"})
    (line,) = finish_lines(tmp_path / "t")
    assert EMAIL not in json.dumps(line)
    assert "[REDACTED:PII_EMAIL]" in line["request"]["prompt"]
    assert "[REDACTED:PII_EMAIL]" in line["response"]["text"]


def test_redact_auto_without_the_extra_says_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in [m for m in sys.modules if m.startswith("dagnam_contracts")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "dagnam_contracts", None)
    with pytest.raises(ImportError, match=r"dagnam-record\[redact\]"):
        dagnam_record.install(dir=str(tmp_path / "t"), redact="auto")
    assert not (tmp_path / "t").exists()


def test_a_failed_reconfiguration_leaves_the_running_recorder_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"))
    for name in [m for m in sys.modules if m.startswith("dagnam_contracts")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "dagnam_contracts", None)
    with pytest.raises(ImportError):
        dagnam_record.install(dir=str(tmp_path / "t"), redact="auto")
    record()
    assert len(finish_lines(tmp_path / "t")) == 1


def test_no_raw_writes_the_norm_only_and_the_line_is_still_valid(
    tmp_path: Path, chat_bodies: Any, validate_line: Any
) -> None:
    request, response = chat_bodies
    dagnam_record.install(dir=str(tmp_path / "t"), no_raw=True)
    record(provider="openai", endpoint="chat.completions", request=request, response=response)
    dagnam_record.uninstall()
    (path,) = (tmp_path / "t").glob("*.jsonl")
    (text,) = path.read_text(encoding="utf-8").splitlines()
    validate_line(text)
    line = json.loads(text)
    assert "request" not in line
    assert "response" not in line
    assert line["norm_v"] == 1
    assert line["norm"]["response"] == "Hello"
    assert set(line) == {
        "v",
        "trace_id",
        "ts",
        "provider",
        "endpoint",
        "latency_ms",
        "status",
        "norm",
        "norm_v",
    }


def test_no_raw_for_a_provider_without_a_norm_is_counted_not_written(tmp_path: Path) -> None:
    dagnam_record.install(dir=str(tmp_path / "t"), no_raw=True)
    record()  # provider "other" has no norm, so there would be nothing to keep
    assert finish_lines(tmp_path / "t") == []
    assert saved_counts(tmp_path / "t")["dropped_error"] == 1
