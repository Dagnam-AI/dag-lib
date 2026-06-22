from __future__ import annotations

from pathlib import Path

from dagnam.data.loaders.system.decoders import get_decoder


def test_system_tabular_decoder_reads_declared_csv_columns(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("price,rooms\n10,2\n20,3\n", encoding="utf-8")

    store = get_decoder("tabular").decode(
        tmp_path,
        {"price": {"column": "price"}, "rooms": {"column": "rooms"}},
        "train",
    )

    assert len(store) == 2
    assert store.column("price")[1] == 20
    assert store.column("rooms")[0] == 2
