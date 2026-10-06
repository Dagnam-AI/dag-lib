"""Rows are redacted last: whatever a cut leaves, the uploaded row scans clean, and a second pass changes nothing.

The adversarial text is assembled here from pieces -- nothing in this file is a literal identifier.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

from dagnam_contracts.hygiene import PII_CODES, scan_rows
from dagnam_contracts.prompts import render_chat_prompt
import pytest

from dagnam.audit.derive import _BUILDERS, build_dataset
from dagnam.audit.record import Message, TraceRecord
from dagnam.audit.redact import redact_records, redact_rows


def _luhn_tail(prefix: str) -> str:
    """The check digit that makes ``prefix`` a Luhn-valid number."""
    total = 0
    for place, digit in enumerate(reversed(prefix)):
        value = int(digit) * (2 if place % 2 == 0 else 1)
        total += value - 9 if value > 9 else value
    return str((10 - total % 10) % 10)


def _identifiers() -> dict[str, str]:
    card_prefix = "".join(["4111", "1111", "1111", "111"])
    return {
        "email": "@".join(["m" + "ia.k" + "ing", ".".join(["acme", "org"])]),
        "national_id": "-".join(["123", "45", "6789"]),
        "card": card_prefix + _luhn_tail(card_prefix),
        "phone": " ".join(["+1", "-".join(["415", "555", "0123"])]),
        "secret": "Bearer " + "-".join(["sk", "live", "".join(["AbCdEfGh", "IjKlMnOpQrStUvWx"])]),
    }


IDENTIFIERS = _identifiers()
SYSTEM = "classify"


def _record(turns: tuple[str, ...], response: str = "billing") -> TraceRecord:
    return TraceRecord(
        trace_id="t",
        ts=datetime(2026, 8, 1, tzinfo=UTC),
        model="m",
        system=SYSTEM,
        messages=tuple(Message("user", t) for t in turns),
        response=response,
        response_tool_calls=(),
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
        cost_usd=None,
        session_id=None,
        outcome=None,
        workload_hint=None,
    )


def _findings(rows: list[dict[str, Any]]) -> dict[str, int]:
    found = scan_rows(rows, max_issues=0).counts_by_code
    return {code: n for code, n in found.items() if n}


def _cuts_through(ident: str) -> list[tuple[str, int]]:
    """``(turn, cap)`` pairs for a labeled row whose front cut lands at every offset of ``ident``."""
    pairs = []
    for into in range(1, len(ident)):
        turn = f"note {ident} " + "x" * 40
        rendered = render_chat_prompt([{"role": "user", "content": turn}], system=SYSTEM)
        pairs.append((turn, len(rendered) - rendered.index(ident) - into))
    return pairs


@pytest.mark.parametrize("name", sorted(IDENTIFIERS))
def test_a_front_cut_through_any_part_of_an_identifier_leaves_a_row_that_scans_clean(
    name: str,
) -> None:
    for turn, cap in _cuts_through(IDENTIFIERS[name]):
        rows = build_dataset(
            [_record((turn,))], structure_class="enum_label", max_seq_length=cap
        ).rows

        assert len(rows[0]["input"]) <= cap  # the budget holds: a placeholder is not added after
        assert _findings(rows) == {}, (name, cap)


@pytest.mark.parametrize("name", sorted(IDENTIFIERS))
def test_a_long_last_turn_kept_whole_over_the_cap_scans_clean(name: str) -> None:
    ident = IDENTIFIERS[name]
    old = _record((f"first {ident}", "second", f"{ident} " + "y" * 500), json.dumps({"k": ident}))

    rows = build_dataset([old], structure_class="json_object", max_seq_length=60).rows

    assert _findings(rows) == {}
    assert ident not in json.dumps(rows)
    assert len(rows[0]["messages"]) == 3  # system, the last turn (kept over budget), the answer


def _all_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ident in IDENTIFIERS.values():
        for turn, cap in _cuts_through(ident)[::7]:
            rows += build_dataset(
                [_record((turn,))], structure_class="enum_label", max_seq_length=cap
            ).rows
        rows += build_dataset(
            [_record((f"a {ident}", "b"), ident)], structure_class="free_text", max_seq_length=50
        ).rows
    return rows


def test_redacting_the_derived_rows_again_changes_nothing() -> None:
    rows = _all_rows()
    assert rows

    again, stats = redact_rows(rows)

    assert again == rows
    assert stats.rows_changed == 0
    assert set(stats.counts) == set(PII_CODES)
    assert not any(stats.counts.values())


def test_the_contract_scan_of_every_class_finds_each_identifier_in_the_raw_text() -> None:
    """The adversarial text is real: unredacted, the contract's own scan reads every one of them."""
    for name, ident in IDENTIFIERS.items():
        assert _findings([{"input": f"see {ident} now"}]), name


def test_the_counts_and_the_changed_truths_are_those_of_the_whole_record() -> None:
    ident = IDENTIFIERS["email"]
    records = [_record((f"mail {ident}",), response=ident), _record(("plain",), response="refund")]

    data = build_dataset(records, structure_class="free_text", max_seq_length=4_096)

    assert data.stats["redact"]["counts"]["PII_EMAIL"] == 2  # in the prompt and in the answer
    assert data.stats["redact"]["rows_changed"] == 1
    assert data.stats["redact"]["truths_changed"] == 1


def _glued(first: str, second: str) -> list[tuple[str, int]]:
    """``(turn, cap)`` for two identifiers run together, so a cut can leave a whole one in the tail."""
    turn = f"note {first}1{second} tail"
    rendered = render_chat_prompt([{"role": "user", "content": turn}], system=SYSTEM)
    start = rendered.index(first)
    return [(turn, len(rendered) - start - into) for into in range(0, len(first) + len(second) + 2)]


@pytest.mark.parametrize(
    "pair", [("national_id", "national_id"), ("card", "phone"), ("phone", "card")]
)
def test_text_a_cut_leaves_next_to_an_identifier_scans_clean(pair: tuple[str, str]) -> None:
    for turn, cap in _glued(IDENTIFIERS[pair[0]], IDENTIFIERS[pair[1]]):
        rows = build_dataset(
            [_record((turn,))], structure_class="enum_label", max_seq_length=cap
        ).rows
        assert _findings(rows) == {}, (pair, cap)


def test_a_cut_alone_does_leak_that_text_so_the_last_pass_is_what_removes_it() -> None:
    """Glued digits are not an identifier whole, but the tail a cut leaves is one: only a last pass sees it."""
    ident = IDENTIFIERS["national_id"]
    leaked = []
    for turn, cap in _glued(ident, ident):
        (clean,), _ = redact_records([_record((turn,))])
        row, _ = _BUILDERS["labeled-example"](clean, cap)  # the cut, with no pass after it
        leaked += [cap] if _findings([row]) else []
    assert leaked
