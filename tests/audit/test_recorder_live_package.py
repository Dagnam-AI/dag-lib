"""The released ``dagnam-record`` package writes lines the SDK's recorder reader reads.

``test_recorder_fixture_pins.py`` pins copied bytes; this one runs the installed package (the dev
group holds ``dagnam-record``), so a release that changes its envelope fails here at once. It
goes through the public ``record_call`` with an OpenAI and an Anthropic body, into a temp directory.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import dagnam_record
import pytest

from dagnam.audit.readers import read_traces

PROMPT = "My widget order W-1001 arrived with a cracked lid."
OPENAI = (
    {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Reply with one intent label."},
            {"role": "user", "content": PROMPT},
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "returns"},
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 2, "total_tokens": 42},
    },
)
ANTHROPIC = (
    {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": "Answer in one word.",
        "messages": [{"role": "user", "content": PROMPT}],
    },
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "returns"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 25, "output_tokens": 4},
    },
)


@pytest.fixture
def traces(tmp_path: Path) -> Iterator[Path]:
    directory = tmp_path / "traces"
    dagnam_record.install(dir=str(directory))
    yield directory
    dagnam_record.uninstall()


def test_calls_the_package_recorded_read_back_through_the_sdk(traces: Path) -> None:
    for provider, endpoint, (request, response) in (
        ("openai", "chat.completions", OPENAI),
        ("anthropic", "messages", ANTHROPIC),
    ):
        dagnam_record.record_call(
            provider=provider, endpoint=endpoint, request=request, response=response, latency_ms=7
        )
    dagnam_record.uninstall()  # writes what is queued
    (path,) = traces.glob("*.jsonl")
    records, stats = read_traces(path, source="dagnam")
    rows = sorted(records, key=lambda r: r.model)
    assert [
        (r.model, r.system, r.response, r.prompt_tokens, r.completion_tokens) for r in rows
    ] == [
        ("claude-sonnet-4-5-20250929", "Answer in one word.", "returns", 25, 4),
        ("gpt-4o-mini-2024-07-18", "Reply with one intent label.", "returns", 40, 2),
    ]
    assert [m.content for r in rows for m in r.messages] == [PROMPT, PROMPT]
    assert (stats.rows_kept, stats.rows_malformed) == (2, 0)
