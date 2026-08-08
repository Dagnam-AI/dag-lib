"""The SDK reproduces the backend's per-component param-diagnostics golden
byte-for-byte — the structured-message parity oracle (spec §6). The catalog is
shipped in the generated schema; the SDK renders from it, it never hand-writes
messages."""

from __future__ import annotations

import json
from pathlib import Path

from dagnam._contracts import validate_params

_GOLDEN = json.loads(Path(__file__).with_name("param-diagnostics.json").read_text("utf-8"))


def test_sdk_reproduces_param_diagnostics_golden() -> None:
    for case in _GOLDEN["cases"]:
        errs = validate_params(case["component_id"], case["config"], case["node_id"])
        got = [
            {
                "code": e.code,
                "severity": e.severity,
                "field": e.field,
                "expected": e.expected,
                "got": e.got,
                "fix_hint": e.fix_hint,
                "message": e.message,
            }
            for e in errs
        ]
        assert got == case["diagnostics"], case["component_id"]
