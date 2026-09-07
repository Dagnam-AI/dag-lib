"""The versioned vendor price table and the estimated serving rates."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from dagnam.audit.prices import (
    PRICES_DIR,
    SERVING_RATES,
    PriceRow,
    PriceTable,
    PriceTableError,
    canonical_model_id,
)
from dagnam.audit.thresholds import PRICE_TABLE_STALE_DAYS

ONE_MILLION = 1_000_000


def test_bundled_table_is_the_newest_and_copies_the_vendor_rows() -> None:
    table = PriceTable.load(None)

    assert table.version == "2026-09"
    assert table.as_of == date(2026, 9, 6)
    assert table.rows["gpt-4o-mini"] == PriceRow("gpt-4o-mini", 0.15, 0.60, 0.075, None)
    assert table.rows["gpt-4o"].cheaper_variant == "gpt-4o-mini"
    assert table.rows["claude-sonnet-5"] == PriceRow(
        "claude-sonnet-5", 2.0, 10.0, 0.2, "claude-haiku-4-5"
    )
    assert table.rows["gemini-2.5-pro"].cheaper_variant == "gemini-2.5-flash"
    assert PriceTable.load(PRICES_DIR / "2026-09.json") == table


def test_every_row_is_well_formed_and_every_cheaper_variant_is_a_row() -> None:
    table = PriceTable.load(None)
    raw = json.loads((PRICES_DIR / "2026-09.json").read_text())

    assert set(raw["_sources"]) == {
        "openai",
        "anthropic",
        "google",
        "mistral",
        "deepseek",
        "xai",
        "cohere",
    }
    assert all({"url", "date"} <= set(src) for src in raw["_sources"].values())
    for model, row in table.rows.items():
        assert row.model == model
        assert 0 < row.input_per_m <= row.output_per_m
        assert row.cached_input_per_m is None or 0 < row.cached_input_per_m <= row.input_per_m
        if row.cheaper_variant is not None:
            cheaper = table.rows[row.cheaper_variant]
            assert cheaper.input_per_m < row.input_per_m


@pytest.mark.parametrize(
    ("model", "canonical"),
    [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("GPT-4o-Mini", "gpt-4o-mini"),
        (" openai/gpt-4o-mini ", "gpt-4o-mini"),
        ("anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("openrouter:anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("google/gemini-2.5-flash", "gemini-2.5-flash"),
        ("vertex_ai/gemini-2.5-flash", "gemini-2.5-flash"),
        ("azure/gpt-4o", "gpt-4o"),
        ("bedrock/us.anthropic.claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("eu.anthropic.claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("mistral/mistral-medium-latest", "mistral-medium"),
        ("deepseek/deepseek-v4-pro", "deepseek-v4-pro"),
        ("xai:grok-4.6", "grok-4.6"),
        ("cohere/command-r-plus-08-2024", "command-r-plus-08-2024"),
        ("groq/llama-3.3-70b", "llama-3.3-70b"),
        ("together/qwen", "qwen"),
        ("meta/llama-4", "llama-4"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        ("some-vendor/unlisted-model", "some-vendor/unlisted-model"),
    ],
)
def test_canonical_model_id_strips_only_the_decoration(model: str, canonical: str) -> None:
    assert canonical_model_id(model) == canonical


def test_canonical_model_id_is_the_identity_on_every_row_of_the_table() -> None:
    for model in PriceTable.load(None).rows:
        assert canonical_model_id(model) == model


@pytest.mark.parametrize(
    ("model", "priced_like"),
    [
        ("anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("ANTHROPIC/Claude-Sonnet-4-5", "claude-sonnet-4-5"),
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("mistral-medium-latest", "mistral-medium"),
        ("openrouter:mistral/mistral-large-latest", "mistral-large"),
        # A dated id the table does not list falls back to the undated row...
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("claude-haiku-4-5-2025-10-01", "claude-haiku-4-5"),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929"),
        ("bedrock/anthropic.claude-haiku-4-5:0", "claude-haiku-4-5"),
    ],
)
def test_a_decorated_id_prices_as_its_table_row(model: str, priced_like: str) -> None:
    table = PriceTable.load(None)

    assert table.cost(model, ONE_MILLION, ONE_MILLION) == table.cost(
        priced_like, ONE_MILLION, ONE_MILLION
    )


def test_a_dated_row_keeps_its_own_price_rather_than_the_undated_one() -> None:
    table = PriceTable.load(None)

    # ...but never over a row the vendor prices separately.
    assert table.cost("gpt-4o-2024-05-13", ONE_MILLION, 0) == 5.0
    assert table.cost("gpt-4o", ONE_MILLION, 0) == 2.50


def test_cost_is_per_million_tokens_and_none_for_an_unknown_model() -> None:
    table = PriceTable.load(None)

    assert table.cost("gpt-4o-mini", ONE_MILLION, ONE_MILLION) == pytest.approx(0.75)
    assert table.cost("gpt-4o-mini", 0, 0) == 0.0
    assert table.cost("no-such-model", ONE_MILLION, ONE_MILLION) is None


def test_age_days_counts_from_the_table_date() -> None:
    table = PriceTable(version="2026-09", as_of=date(2026, 9, 6), rows={})

    assert table.age_days(date(2026, 9, 6)) == 0
    assert table.age_days(date(2026, 11, 5)) == PRICE_TABLE_STALE_DAYS == 60


def test_missing_path_is_a_price_table_error(tmp_path: Path) -> None:
    with pytest.raises(PriceTableError, match="no such file"):
        PriceTable.load(tmp_path / "2099-01.json")


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("{", "not valid JSON"),
        ("[]", "must be a JSON object"),
        ('{"version": "x", "as_of": "2026-09-06"}', "rows"),
        ('{"version": 1, "as_of": "2026-09-06", "rows": {}}', "version"),
        ('{"version": "x", "as_of": "yesterday", "rows": {}}', "as_of"),
        ('{"version": "x", "as_of": "2026-09-06", "rows": []}', "rows"),
        ('{"version": "x", "as_of": "2026-09-06", "rows": {"m": 1}}', "row m"),
        (
            '{"version": "x", "as_of": "2026-09-06", "rows": {"m": {"input_per_m": 1}}}',
            "row m",
        ),
        (
            '{"version": "x", "as_of": "2026-09-06", "rows": {"m": {"input_per_m": "1",'
            ' "output_per_m": 2, "cached_input_per_m": null, "cheaper_variant": null}}}',
            "row m",
        ),
        (
            '{"version": "x", "as_of": "2026-09-06", "rows": {"m": {"input_per_m": 1,'
            ' "output_per_m": 2, "cached_input_per_m": null, "cheaper_variant": 3}}}',
            "row m",
        ),
    ],
)
def test_malformed_table_names_the_defect(tmp_path: Path, text: str, why: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(PriceTableError, match=why):
        PriceTable.load(path)


def test_optional_row_fields_are_optional(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(
        '{"version": "x", "as_of": "2026-09-06", "rows": {"m": {"input_per_m": 1,'
        ' "output_per_m": 2, "cached_input_per_m": 0.5, "cheaper_variant": "m"}}}'
    )
    assert PriceTable.load(path).rows["m"] == PriceRow("m", 1.0, 2.0, 0.5, "m")


def test_serving_rates_are_the_two_estimated_rows() -> None:
    assert set(SERVING_RATES) == {"cpu-classifier", "gpu-small-llm"}
    assert SERVING_RATES["cpu-classifier"]["usd_per_1k_requests"] > 0
    assert SERVING_RATES["gpu-small-llm"]["usd_per_m_output_tokens"] > 0
    for rate in SERVING_RATES.values():
        assert rate["basis"] == "estimated"
        assert rate["assumptions"]
