"""The shared recorder: plain-JSON conversion, fail-open counting, the kill switch and emit."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from typing_extensions import override

from dagnam_record._envelope import LineTooLargeError
from dagnam_record._recorder import Recorder, media_omitted, recording_disabled, to_plain


class NotGiven:
    """Stands in for the SDKs' "argument not passed" sentinel (found by class name)."""


class Pydantic2:
    """A pydantic v2 model."""

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Dump."""
        return {"how": "v2", "kwargs": sorted(kwargs)}


class OldPydantic2:
    """A pydantic v2 model whose ``model_dump`` lacks some keywords."""

    def model_dump(self, *, mode: str) -> Dict[str, Any]:
        """Dump."""
        return {"how": "v2 without exclude_unset", "mode": mode}


class ToDict:
    """An SDK object that only has ``to_dict``."""

    def to_dict(self) -> Dict[str, Any]:
        """Dump."""
        return {"how": "to_dict"}


class Pydantic1:
    """A pydantic v1 model."""

    def dict(self) -> Dict[str, Any]:
        """Dump."""
        return {"how": "dict"}


class Unreadable:
    """An object whose dump raises and whose repr holds a secret."""

    def model_dump(self, **kwargs: Any) -> Any:
        """Raise."""
        raise RuntimeError("cannot dump")

    @override
    def __repr__(self) -> str:
        return "Unreadable(secret=sk-must-not-leak)"


class NotADict:
    """An object whose dump is not a dict."""

    def model_dump(self, **kwargs: Any) -> Any:
        """Not a dict."""
        return ["a list"]


class NotCallable:
    """An object whose ``model_dump`` is not a method."""

    model_dump = 5


class Color(Enum):
    """A plain enum."""

    RED = "red"


class Mode(str, Enum):
    """A string enum."""

    FAST = "fast"


def test_json_types_pass_through_and_containers_are_copied() -> None:
    source: Dict[Any, Any] = {"a": [1, 2.5, True, None, "x"], "b": {"c": ("t", "u")}, 3: "int key"}
    plain = to_plain(source)
    assert plain == {"a": [1, 2.5, True, None, "x"], "b": {"c": ["t", "u"]}, "3": "int key"}
    assert plain["a"] is not source["a"]


def test_not_given_values_are_dropped_from_a_dict() -> None:
    assert to_plain({"keep": 1, "gone": NotGiven()}) == {"keep": 1}


def test_bytes_become_a_size_marker_and_dates_text_and_enums_their_value() -> None:
    assert to_plain(b"abcd") == {"type": "media_omitted", "bytes": 4}
    assert to_plain(bytearray(b"ab")) == {"type": "media_omitted", "bytes": 2}
    assert to_plain(date(2026, 10, 23)) == "2026-10-23"
    assert (
        to_plain(datetime(2026, 10, 23, 1, 2, 3, tzinfo=timezone.utc))
        == "2026-10-23T01:02:03+00:00"
    )
    assert to_plain(Color.RED) == "red"
    assert to_plain(Mode.FAST) == "fast"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Pydantic2(), {"how": "v2", "kwargs": ["exclude_unset", "mode", "warnings"]}),
        (OldPydantic2(), {"how": "v2 without exclude_unset", "mode": "json"}),
        (ToDict(), {"how": "to_dict"}),
        (Pydantic1(), {"how": "dict"}),
    ],
)
def test_sdk_objects_become_dicts_whichever_way_they_can_be_dumped(
    value: Any, expected: Dict[str, Any]
) -> None:
    assert to_plain({"x": value}) == {"x": expected}


@pytest.mark.parametrize("value", [Unreadable(), NotADict(), NotCallable(), object()])
def test_anything_else_is_named_by_its_class_and_never_printed(value: Any) -> None:
    assert to_plain(value) == f"<{type(value).__name__}>"
    assert "sk-must-not-leak" not in json.dumps(to_plain([value]))


def test_absurdly_deep_data_is_cut_off_instead_of_recursing_forever() -> None:
    deep: Any = "end"
    for _ in range(200):
        deep = [deep]
    plain = to_plain(deep)
    depth = 0
    while isinstance(plain, list):
        plain = plain[0]
        depth += 1
    assert plain == "<too deep>"
    assert depth < 100
    cycle: List[Any] = []
    cycle.append(cycle)
    assert "<too deep>" in json.dumps(to_plain(cycle))


def test_media_omitted_counts_utf8_bytes() -> None:
    assert media_omitted("é") == {"type": "media_omitted", "bytes": 2}


def test_the_kill_switch_is_exactly_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAGNAM_RECORD", raising=False)
    assert recording_disabled() is False
    monkeypatch.setenv("DAGNAM_RECORD", "1")
    assert recording_disabled() is False
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    assert recording_disabled() is True


def test_fail_open_runs_a_step_and_counts_what_goes_wrong(make: Any) -> None:
    writer = make()
    recorder = Recorder(writer)
    ran: List[int] = []
    assert recorder.fail_open(ran.append, 1) is True
    assert ran == [1]

    def too_large() -> None:
        raise LineTooLargeError("big")

    def broken() -> None:
        raise RuntimeError("bug")

    assert recorder.fail_open(too_large) is False
    assert recorder.fail_open(broken) is False
    assert recorder.fail_open(broken) is False
    assert (writer.counts["dropped_large"], writer.counts["dropped_error"]) == (1, 2)


def test_emit_builds_one_line_and_queues_it(make: Any, tmp_path: Path, validate_line: Any) -> None:
    writer = make()
    request: Dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
    response: Dict[str, Any] = {"choices": [{"message": {"content": "Yo"}}]}
    Recorder(writer).emit("openai", "chat.completions", request, response, status=200, started=0.0)
    writer.drain()
    (path,) = (tmp_path / "traces").glob("*.jsonl")
    (raw,) = path.read_text(encoding="utf-8").splitlines()
    validate_line(raw)
    line = json.loads(raw)
    assert (line["request"], line["response"], line["status"]) == (request, response, 200)
    assert line["norm"]["response"] == "Yo"
    assert line["latency_ms"] > 0  # measured from the perf_counter start


def test_emit_without_raw_keeps_only_the_norm_and_a_missing_status_is_599(
    make: Any, tmp_path: Path
) -> None:
    writer = make()
    Recorder(writer, no_raw=True).emit(
        "openai", "responses", {"input": "Hi"}, {"output": []}, status=None, started=0.0
    )
    writer.drain()
    (path,) = (tmp_path / "traces").glob("*.jsonl")
    line = json.loads(path.read_text(encoding="utf-8"))
    assert "request" not in line
    assert "response" not in line
    assert line["status"] == 599


def test_emit_refuses_an_unknown_provider_with_a_value_error(make: Any) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        Recorder(make()).emit("acme", "chat.completions", {}, {}, status=200, started=0.0)
