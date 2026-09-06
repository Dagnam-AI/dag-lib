"""Redaction: every contract PII class, counts per class, the pass list."""

from __future__ import annotations

from dagnam_contracts.hygiene import PII_CODES

from dagnam.audit.redact import PII_POLICY, RedactStats, redact_rows


def test_policy_redacts_every_class_the_contract_knows() -> None:
    assert dict.fromkeys(PII_CODES, "redact") == PII_POLICY


def test_redaction_counts_and_pass_list_are_reported() -> None:
    rows, stats = redact_rows([{"input": "mail me at a@b.co", "label": "x"}])

    assert rows[0] == {"input": "mail me at [REDACTED:PII_EMAIL]", "label": "x"}
    assert stats.counts["PII_EMAIL"] == 1
    assert "PII_PHONE" in stats.pass_list
    assert stats.pass_list == PII_CODES
    assert stats.rows_changed == 1


def test_nested_chat_messages_are_redacted_and_rows_never_dropped() -> None:
    rows = [
        {"messages": [{"role": "user", "content": "call +1 415-555-0123 or a@b.co"}]},
        {"messages": [{"role": "user", "content": "nothing here"}]},
    ]
    redacted, stats = redact_rows(rows)

    assert len(redacted) == 2
    assert redacted[0]["messages"][0]["content"] == (
        "call [REDACTED:PII_PHONE] or [REDACTED:PII_EMAIL]"
    )
    assert redacted[1] == rows[1]
    assert stats == RedactStats(
        counts={"PII_EMAIL": 1, "PII_PHONE": 1, "PII_PAYMENT_CARD": 0, "PII_NATIONAL_ID": 0},
        pass_list=PII_CODES,
        rows_changed=1,
    )


def test_input_rows_are_not_mutated() -> None:
    rows = [{"input": "a@b.co", "label": "x"}]
    redact_rows(rows)
    assert rows == [{"input": "a@b.co", "label": "x"}]
