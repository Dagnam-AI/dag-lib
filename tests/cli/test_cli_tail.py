"""Patch the remaining CLI / loader / dataset gaps."""

from __future__ import annotations
from pathlib import Path
from tests.typing_helpers import JsonObject, PytestMonkeyPatch, StrCapture


import polars as pl
import pytest

from dagnam.cli import main as cli_main
from dagnam.cli.common import human_size
from dagnam.data.loaders.csv import detect_label_column, split_by_roles

# ---------------------------------------------------------------- cli/common


def testhuman_size_petabytes() -> None:
    assert human_size(2 * 1024**5) == "2.0 PB"


# ---------------------------------------------------------------- cli/cache edge cases


def test_cache_list_skips_files_and_bad_meta(tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    # A file at the top level — should be skipped (not a dir).
    (cache_dir / "stray.txt").write_text("noise")
    # A dataset dir with a malformed meta.json — should be tolerated.
    bad = cache_dir / "ds-bad"
    bad.mkdir()
    (bad / "meta.json").write_text("not json {{{")
    (bad / "data").write_bytes(b"x")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)
    monkeypatch.setattr("sys.argv", ["dagnam", "cache", "list"])
    cli_main()
    out = capsys.readouterr().out
    assert "ds-bad" in out
    assert "stray.txt" not in out


def test_cache_clear_when_already_empty(tmp_path: Path, monkeypatch: PytestMonkeyPatch, capsys: StrCapture) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", missing)
    monkeypatch.setattr("sys.argv", ["dagnam", "cache", "clear"])
    cli_main()
    assert "already empty" in capsys.readouterr().out


# ---------------------------------------------------------------- cli/dataset auth error in info


def test_dataset_info_autherror_exits(tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
    from pathlib import Path

    monkeypatch.delenv("DAGNAM_API_KEY", raising=False)
    monkeypatch.setattr("dagnam._core.config.CONFIG_FILE", Path(tmp_path) / "missing.json")
    monkeypatch.setattr("sys.argv", ["dagnam", "dataset", "info", "ds-1"])
    with pytest.raises(SystemExit):
        cli_main()


# ---------------------------------------------------------------- csv loader role branches


def testdetect_label_column_uses_roles_priority() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    assert detect_label_column(df, None, column_roles={"b": "target"}) == "b"
    assert detect_label_column(df, None, column_roles={"b": "label"}) == "b"


def testsplit_by_roles_raises_when_no_target_column() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="target column"):
        split_by_roles(df, {"a": "feature", "b": "ignore"})


def test_create_pytorch_loader_with_explicit_column_roles(tmp_path: Path, sample_metadata: JsonObject) -> None:
    """Exercises the create_pytorch_loader branch that calls split_by_roles."""
    from dagnam.data.dataset import DagnamDataset
    from dagnam.data.loaders.csv import create_pytorch_loader

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,label\n1.0,2.0,a\n3.0,4.0,b\n5.0,6.0,a\n7.0,8.0,b\n9.0,10.0,a\n")

    meta = dict(sample_metadata)
    meta["filename"] = "data.csv"
    meta["format"] = "csv"
    meta["dataset_type"] = "tabular"
    meta["class_names"] = ["a", "b"]
    ds = DagnamDataset(meta, tmp_path)
    loader = create_pytorch_loader(
        ds,
        split="train",
        batch_size=2,
        num_workers=0,
        shuffle=False,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
        column_roles={"x": "feature", "y": "feature", "label": "target"},
    )
    assert loader is not None
