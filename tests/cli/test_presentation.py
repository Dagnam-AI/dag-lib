from pathlib import Path

from dagnam.cli.presentation import Column, emit_result, pagination_footer, render_table


def test_render_table_truncates_long_cells_without_shifting_columns() -> None:
    table = render_table(
        [
            Column("NAME", "name", max_width=8),
            Column("COUNT", "count", max_width=5, align="right"),
        ],
        [{"name": "image_folder_dataset", "count": 12}],
    )

    assert "image..." in table
    assert table.splitlines()[2].endswith("   12")


def test_render_table_supports_empty_rows() -> None:
    assert render_table([Column("NAME", "name", max_width=8)], []) == "NAME\n----"


def test_pagination_footer_normalizes_empty_page_count() -> None:
    assert pagination_footer({"page": 1, "pages": 0, "total": 0, "items": []}) == (
        "Page 1 of 1 - showing 0 of 0"
    )


def test_emit_result_writes_json_file_and_human_stdout(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "result.json"

    emit_result(
        {"items": [{"id": "one"}]},
        output=output,
        json_stdout=False,
        render_human=lambda _result: "one row",
    )

    assert (
        output.read_text(encoding="utf-8")
        == '{\n  "items": [\n    {\n      "id": "one"\n    }\n  ]\n}\n'
    )
    assert capsys.readouterr().out == "one row\n"


def test_sanitize_terminal_text_strips_escape_sequences() -> None:
    from dagnam.cli.presentation import sanitize_terminal_text

    # OSC 8 hyperlink spoof + a CSI sequence + a bare ESC, all neutralised.
    hostile = "\x1b]8;;http://evil.example\x07Model\x1b]8;;\x07\x1b[31m\x9b2J"
    cleaned = sanitize_terminal_text(hostile)
    assert "\x1b" not in cleaned
    assert "\x9b" not in cleaned
    assert "\x07" not in cleaned
    assert "Model" in cleaned  # printable content preserved


def test_sanitize_terminal_text_keeps_tabs_and_newlines() -> None:
    from dagnam.cli.presentation import sanitize_terminal_text

    assert sanitize_terminal_text("a\tb\nc") == "a\tb\nc"


def test_render_table_neutralises_hostile_cell_name() -> None:
    # A hostile dataset/hub name must not reach the TTY as a live escape.
    table = render_table(
        [Column("NAME", "name", max_width=40)],
        [{"name": "\x1b]0;pwned\x07official"}],
    )
    assert "\x1b" not in table
    assert "official" in table
