"""Template normalization masks the variable parts of a system prompt; the hash is the workload id."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from dagnam.audit.normalize import UNSTRUCTURED, normalize_template, template_hash


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Order 12345 for user@x.com", "Order 99 for me@y.org"),
        ("Ticket {{id}} at 2026-09-02T10:00:00Z", "Ticket {{other}} at 2025-01-01T00:00:00Z"),
        ("Fill {product, order_id}", "Fill {name}"),
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
        ("Fill {product, order_id}", "Fill {VAR}"),
        ("go to https://a.example/x?y=1 now", "go to <URL> now"),
        ("id 123e4567-e89b-12d3-a456-426614174000.", "id <UUID>."),
        ('Say "this quoted span is longer than 24 chars" now', "Say <QUOTED> now"),
        ('Say "short quote" now', 'Say "short quote" now'),
        ("a  b\n\tc", "a b c"),
        ("", ""),
        # Second-pass cases: a masked span joins two short quotes into one long
        # one; a newline inside braces only becomes a placeholder once collapsed.
        ('"' + "a" * 20 + '" ' + "b" * 26 + ' "' + "c" * 14 + '"', "<QUOTED>"),
        ("Fill {product,\n order_id}", "Fill {VAR}"),
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
