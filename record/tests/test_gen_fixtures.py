"""The committed recorder fixtures are what ``gen_fixtures.py`` writes today, byte for byte.

This runs the real ``openai`` and ``anthropic`` clients against the mock server, so it also proves
the recorder still records those SDKs the way the fixtures say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import gen_fixtures
import openai
import pytest

LOCKED = {"openai": openai.__version__, "anthropic": anthropic.__version__} == (
    gen_fixtures.SDK_VERSIONS
)


def lines(directory: Path, name: str) -> list[str]:
    return (directory / name).read_text(encoding="utf-8").splitlines()


@pytest.mark.skipif(
    not LOCKED, reason="the fixtures are generated with the locked SDK versions (SDK_VERSIONS)"
)
@pytest.mark.skipif(
    not gen_fixtures.FIXTURES.is_dir(), reason="only a dag-lib checkout holds the fixtures"
)
def test_the_committed_fixtures_are_what_the_generator_writes_and_it_is_deterministic(
    fresh_state: None, tmp_path: Path
) -> None:
    first = gen_fixtures.generate(tmp_path / "first")
    again = gen_fixtures.generate(tmp_path / "again")
    assert first == again
    assert set(first) == {path.name for path in gen_fixtures.FIXTURES.glob("*.jsonl")} - {
        "generic_sample.jsonl",
        "langfuse_sample.jsonl",
        "langsmith_sample.jsonl",
        "openai_sample.jsonl",
    }
    for name, expected in first.items():
        assert lines(gen_fixtures.FIXTURES, name) == expected, name


def test_main_writes_every_file_to_the_directory_it_is_given(
    fresh_state: None, tmp_path: Path, capsys: Any
) -> None:
    assert gen_fixtures.main([str(tmp_path / "out")]) == 0
    written = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert written == sorted([*(spec.name for spec in gen_fixtures.SPECS), gen_fixtures.BATCH])
    assert "recorder_openai_chat.jsonl: 13 lines" in capsys.readouterr().out
    first = json.loads(lines(tmp_path / "out", "recorder_openai_chat.jsonl")[0])
    assert (first["trace_id"], first["ts"], first["latency_ms"]) == (
        "rec_chat_1",
        "2026-10-12T00:00:01.000Z",
        301.0,
    )


def test_the_generator_refuses_to_run_with_the_recorder_switched_off(
    fresh_state: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAGNAM_RECORD", "0")
    with pytest.raises(RuntimeError, match="patched no SDK"):
        gen_fixtures.generate(tmp_path)
