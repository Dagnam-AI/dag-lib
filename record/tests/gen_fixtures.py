"""Regenerate the recorder fixtures in ``tests/audit/fixtures`` from real recorded calls.

A dev script, not a test (its name does not start with ``test_``). It runs the real ``openai`` and
``anthropic`` clients against the local mock server with ``dagnam_record.install()`` writing to a
temporary directory, then copies the lines the recorder wrote. Run it from the repository root:

    HOME="$(mktemp -d)" PYTHONPATH=record uv run python record/tests/gen_fixtures.py [OUT_DIR]

Only three fields of each line are rewritten, so a second run gives the same bytes: ``trace_id``
(``rec_<tag>_<n>``), ``ts`` and ``latency_ms`` (both counted up from a fixed base). The bodies and
the ``norm`` are exactly what the recorder wrote. The output depends on the installed ``openai`` and
``anthropic`` (see ``SDK_VERSIONS``), so regenerate and review the diff when either is bumped.

It also writes ``openai_roundtrip_batch.jsonl``: the calls of ``recorder_openai_chat.jsonl`` as an
OpenAI Batch export, with the request as the mock server received it on the wire (media in full)
and the response the server sent, for the round trip through the OpenAI reader.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from _fixture_env import CHAT, Env
import _fixture_scenarios as openai_calls
import _fixture_scenarios_more as other_calls
from _mock_server import MockServer
import anthropic
import httpx
import openai

import dagnam_record

EPOCH = 1_791_763_200  # 2026-10-12T00:00:00Z
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "audit" / "fixtures"
BATCH = "openai_roundtrip_batch.jsonl"
SDK_VERSIONS = {"openai": "2.54.0", "anthropic": "0.125.0"}
"""The SDK versions the committed fixtures were generated with (the locked ones)."""
API_KEY = "sk-fixture-not-a-real-key"


class Spec(NamedTuple):
    """One output file: its calls, id tag, time and latency bases, and whether it is norm only."""

    name: str
    tag: str
    offset: int
    latency: int
    calls: Callable[[Env], None]
    no_raw: bool = False


SPECS = (
    Spec("recorder_openai_chat.jsonl", "chat", 0, 300, openai_calls.chat_file),
    Spec("recorder_openai_responses.jsonl", "resp", 100, 410, openai_calls.responses_file),
    Spec("recorder_anthropic.jsonl", "anth", 200, 500, other_calls.anthropic_file),
    Spec("recorder_norm_only.jsonl", "norm", 300, 200, other_calls.norm_only_file, no_raw=True),
    Spec("recorder_mixed_bad.jsonl", "mix", 400, 90, other_calls.mixed_bad_file),
    Spec("recorder_openai_stream.jsonl", "ostream", 500, 700, openai_calls.openai_stream_file),
    Spec("recorder_openai_parse.jsonl", "parse", 600, 800, openai_calls.parse_file),
    Spec("recorder_anthropic_stream.jsonl", "astream", 700, 900, other_calls.anthropic_stream_file),
    Spec("recorder_media.jsonl", "media", 800, 150, other_calls.media_file),
    Spec("recorder_other.jsonl", "other", 900, 50, other_calls.other_file),
)


def stamp(seconds: int) -> str:
    """The recorder's own timestamp format for a whole second."""
    text = datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="milliseconds")
    return text.replace("+00:00", "Z")


def normalise(spec: Spec, written: List[str]) -> List[str]:
    """The lines with the three run-dependent fields replaced by fixed, call-numbered ones."""
    lines: List[str] = []
    for n, line in enumerate(written, 1):
        body = json.loads(line)
        body["trace_id"] = f"rec_{spec.tag}_{n}"
        body["ts"] = stamp(EPOCH + spec.offset + n)
        body["latency_ms"] = float(spec.latency + n)
        lines.append(json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    return lines


def record(spec: Spec, env: Env, work: Path) -> List[str]:
    """Make the calls of one file under a fresh recorder and return the lines it wrote."""
    directory = work / spec.tag
    if not dagnam_record.install(dir=directory, no_raw=spec.no_raw):
        raise RuntimeError("the recorder patched no SDK (is DAGNAM_RECORD off?)")
    env.begin(EPOCH + spec.offset)
    try:
        spec.calls(env)
    finally:
        dagnam_record.uninstall()
    written: List[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        written.extend(path.read_text(encoding="utf-8").splitlines())
    return normalise(spec, written)


def batch_lines(chat: List[str], sent: List[Any]) -> List[str]:
    """The chat file's calls as OpenAI Batch rows: the request as sent, the response as received."""
    rows: List[str] = []
    for n, (line, body) in enumerate(zip(chat, sent), 1):
        call = json.loads(line)
        row = {
            "custom_id": f"call-{n}",
            "method": "POST",
            "url": CHAT,
            "body": body,
            "id": f"batch_req_{n:04d}",
            "response": {
                "status_code": 200,
                "request_id": f"req_{n:04d}",
                "body": call["response"],
            },
            "error": None,
            "latency_ms": call["latency_ms"],
        }
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return rows


def generate(work: Path) -> Dict[str, List[str]]:
    """Every fixture file's lines, by file name; the recorder writes under ``work``."""
    server = MockServer()
    http = httpx.Client(trust_env=False)
    env = Env(
        server,
        openai.OpenAI(api_key=API_KEY, base_url=server.url, http_client=http, max_retries=0),
        anthropic.Anthropic(api_key=API_KEY, base_url=server.root, http_client=http, max_retries=0),
    )
    files: Dict[str, List[str]] = {}
    sent: Optional[List[Any]] = None
    try:
        for spec in SPECS:
            seen = len(server.seen)
            files[spec.name] = record(spec, env, work)
            if spec.tag == "chat":
                sent = [s.body for s in server.seen[seen:] if s.path == CHAT]
    finally:
        http.close()
        server.close()
    assert sent is not None
    files[BATCH] = batch_lines(files["recorder_openai_chat.jsonl"], sent)
    return files


def main(argv: List[str]) -> int:
    """Write every fixture into ``argv[0]`` (default: the repository's fixture directory)."""
    out = Path(argv[0]) if argv else FIXTURES
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        files = generate(Path(work))
    for name, lines in files.items():
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        sys.stdout.write(f"{name}: {len(lines)} lines\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
