"""The committed recorder fixtures are what ``gen_fixtures.py`` writes today, byte for byte.

This runs the real ``openai`` and ``anthropic`` clients against the mock server, so it also proves
the recorder still records those SDKs the way the fixtures say.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic
import gen_fixtures
import openai
import pytest

MATRIX = "DAGNAM_RECORD_SDK_MATRIX"
"""Set to 1 by a job that runs the suite against other SDK versions on purpose (it opts out)."""


def lines(directory: Path, name: str) -> list[str]:
    return (directory / name).read_text(encoding="utf-8").splitlines()


def test_the_committed_fixtures_are_what_the_generator_writes_and_it_is_deterministic(
    fresh_state: None, tmp_path: Path
) -> None:
    if os.environ.get(MATRIX) == "1":
        pytest.skip(
            f"{MATRIX}=1: this job runs other SDK versions than the fixtures were made with"
        )
    gen_fixtures.check_environment()
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


def test_the_environment_check_passes_for_the_locked_sdks_and_a_checkout() -> None:
    if os.environ.get(MATRIX) == "1":
        pytest.skip(f"{MATRIX}=1")
    gen_fixtures.check_environment()
    assert gen_fixtures.installed_versions() == {
        "openai": openai.__version__,
        "anthropic": anthropic.__version__,
    }


def test_other_sdk_versions_fail_the_check_with_the_versions_and_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gen_fixtures, "SDK_VERSIONS", {"openai": "0.0.1", "anthropic": "0.0.2"})
    with pytest.raises(RuntimeError) as caught:
        gen_fixtures.check_environment()
    message = str(caught.value)
    assert "openai 0.0.1" in message
    assert f"openai {openai.__version__}" in message
    assert "anthropic 0.0.2" in message
    assert "SDK_VERSIONS" in message
    assert "DAGNAM_RECORD_SDK_MATRIX=1" in message


def test_a_missing_fixture_directory_fails_the_check_with_where_it_looked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gen_fixtures, "FIXTURES", tmp_path / "absent")
    with pytest.raises(RuntimeError, match=r"absent.*DAGNAM_RECORD_SDK_MATRIX=1"):
        gen_fixtures.check_environment()


def test_main_writes_every_file_to_the_directory_it_is_given(
    fresh_state: None, tmp_path: Path, capsys: Any
) -> None:
    assert gen_fixtures.main([str(tmp_path / "out")]) == 0
    written = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert written == sorted([*(spec.name for spec in gen_fixtures.SPECS), gen_fixtures.BATCH])
    assert "recorder_openai_chat.jsonl: 14 lines" in capsys.readouterr().out
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
