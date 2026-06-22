from __future__ import annotations

from pathlib import Path

from dagnam.data.loaders.system.decoders import get_decoder


def test_system_text_decoder_reads_declared_corpus_file_lines(tmp_path: Path) -> None:
    (tmp_path / "wiki.train.tokens").write_text("alpha\nbeta\n\ngamma\n", encoding="utf-8")

    store = get_decoder("text").decode(tmp_path, {"text": {"file": "wiki.train.tokens"}}, "train")

    assert len(store) == 3
    assert str(store.column("text")[0]) == "alpha"
    assert str(store.column("text")[2]) == "gamma"
