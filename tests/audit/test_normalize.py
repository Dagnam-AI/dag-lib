"""Template normalization masks the variable parts of a system prompt; the hash is the workload id."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings, strategies as st
import pytest

from dagnam.audit.normalize import UNSTRUCTURED, normalize_template, template_hash


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Order 12345 for user@x.com", "Order 99 for me@y.org"),
        ("Ticket {{id}} at 2026-09-02T10:00:00Z", "Ticket {{other}} at 2025-01-01T00:00:00Z"),
        ("Fill {product, order_id}", 'Fill {"product": str, "order_id": int}'),
        ("Customer {'name': 'Maria', 'tier': 'gold'}", "Customer {'name': 'Bo', 'tier': 'new'}"),
        ("Fill {Maria Garcia, 12 Elm St}", "Fill {Bo Li, 7 Oak Ave}"),
        ("See https://a.example/x?y=1", "See http://b.example"),
        (
            "Trace 123e4567-e89b-12d3-a456-426614174000",
            "Trace ffffffff-ffff-4fff-8fff-ffffffffffff",
        ),
        (
            'Say "this quoted span is longer than 24 chars"',
            'Say "another one that is also over 24"',
        ),
        ("Due 2026-09-02", "Due 2020-01-31 12:00"),
        ("Total 1,234.56 units", "Total 7 units"),
        ("a  b\n\tc", " a b c "),
    ],
)
def test_normalize_template_masks_variables(a: str, b: str) -> None:
    assert normalize_template(a) == normalize_template(b)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Order 12345 for user@x.com", "Order <NUM> for <EMAIL>"),
        ("Ticket {{id}} at 2026-09-02T10:00:00Z", "Ticket {{VAR}} at <DATE>"),
        ("Fill {product, order_id}", "Fill {product,order_id}"),
        ('Respond with {"product": str, "due": date}', "Respond with {product,due}"),
        ("Fill {Maria Garcia, 12 Elm St}", "Fill {VAR}"),
        ("Today is Sunday, September 27, 2026.", "Today is <DAY>, <MONTH> <NUM>, <NUM>."),
        ("Due 1st Oct at 10:30 AM", "Due <NUM> <MONTH> at <TIME>"),
        ("go to https://a.example/x?y=1 now", "go to <URL> now"),
        ("id 123e4567-e89b-12d3-a456-426614174000.", "id <UUID>."),
        ('Say "this quoted span is longer than 24 chars" now', "Say <QUOTED> now"),
        ('Say "short quote" now', 'Say "short quote" now'),
        ("a  b\n\tc", "a b c"),
        ("", ""),
        # Second-pass cases: a masked span joins two short quotes into one long
        # one; a newline inside braces only becomes a placeholder once collapsed.
        ('"' + "a" * 20 + '" ' + "b" * 26 + ' "' + "c" * 14 + '"', "<QUOTED>"),
        ("Fill {product,\n order_id}", "Fill {product,order_id}"),
    ],
)
def test_each_rule(raw: str, expected: str) -> None:
    assert normalize_template(raw) == expected


def test_prompts_that_differ_in_wording_stay_apart() -> None:
    assert normalize_template("Reply with an intent label") != normalize_template(
        "Reply with an urgency label"
    )


@given(st.text())
@settings(max_examples=300, deadline=None)
def test_normalize_template_is_idempotent(text: str) -> None:
    once = normalize_template(text)
    assert normalize_template(once) == once


def test_template_hash_is_16_hex_and_ignores_variables() -> None:
    digest = template_hash("Order 12345")
    assert len(digest) == 16
    assert int(digest, 16) >= 0
    assert digest == template_hash("Order 6")
    assert digest != template_hash("Order 6 today")


@pytest.mark.parametrize("system", [None, "", "   \n"])
def test_template_hash_is_unstructured_without_a_prompt(system: str | None) -> None:
    assert template_hash(system) == UNSTRUCTURED == "unstructured"


def test_template_hash_is_stable_across_processes() -> None:
    # blake2b of the template, never ``hash()``: the digest below was recorded
    # once and must hold in every interpreter, whatever PYTHONHASHSEED says.
    assert template_hash("Label the ticket 42") == "cc544e204cd9d6b3"


def test_a_human_readable_date_does_not_split_a_workload() -> None:
    # One urgency workload whose prompt opens "Today is Sunday, September 27, 2026."
    # was reported as 14 workloads (weekday x month), every one too small to audit.
    start = datetime(2026, 8, 20, tzinfo=UTC)
    hashes = {
        template_hash(
            f"Today is {start + timedelta(minutes=30 * i):%A, %B %d, %Y}. It is"
            f" {start + timedelta(minutes=30 * i):%I:%M %p}. Classify the ticket's urgency."
        )
        for i in range(1_400)
    }
    assert len(hashes) == 1
    assert normalize_template("Mon, 27 Sep 2026") == normalize_template("Fri, 3 Oct 2026")
    assert normalize_template("the 1st of May") == normalize_template("the 22nd of June")
    assert normalize_template("It may rain") == "It may rain"  # a month name alone is prose


def test_one_line_schemas_keep_different_extraction_tasks_apart() -> None:
    # Two extraction prompts that differ only in their one-line schema collapsed
    # into one workload, classified short_span and reported a candidate.
    products = template_hash(
        'Extract fields. Respond with {"product": str, "sentiment": str} only.'
    )
    invoices = template_hash(
        'Extract fields. Respond with {"invoice_total": float, "currency": str, "due": date} only.'
    )
    assert products != invoices
    assert template_hash("Respond with {product, sentiment}") != template_hash(
        "Respond with {invoice_total, currency, due}"
    )
