"""CLI cache subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.typing_helpers import CliRunner, PytestMonkeyPatch, StrCapture


# ---------------------------------------------------------------- cache


def test_cache_clear(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    (cache_dir / "ds-1").mkdir()
    (cache_dir / "ds-1" / "data").write_text("x")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)
    run_cli(["cache", "clear"])
    out = capsys.readouterr().out
    assert "Cleared" in out or "cleared" in out.lower()


def test_cache_clear_dry_run_keeps_cache(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    entry = cache_dir / "ds-1"
    entry.mkdir(parents=True)
    (entry / "data").write_text("x")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)

    run_cli(["cache", "clear", "--dry-run"])

    assert entry.exists()
    assert "Would clear" in capsys.readouterr().out


def test_cache_clear_dataset_id_only_removes_target(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    (cache_dir / "ds-1").mkdir(parents=True)
    (cache_dir / "ds-1" / "data").write_text("x")
    (cache_dir / "ds-2").mkdir()
    (cache_dir / "ds-2" / "data").write_text("y")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)

    run_cli(["cache", "clear", "--dataset-id", "ds-1"])

    assert not (cache_dir / "ds-1").exists()
    assert (cache_dir / "ds-2").exists()
    assert "ds-1" in capsys.readouterr().out


def test_cache_list_with_entries(
    run_cli: CliRunner, capsys: StrCapture, tmp_path: Path, monkeypatch: PytestMonkeyPatch
) -> None:
    cache_dir = tmp_path / "datasets"
    cache_dir.mkdir()
    sub = cache_dir / "ds-1"
    sub.mkdir()
    (sub / "data").write_bytes(b"hi")
    monkeypatch.setattr("dagnam.data.cache.DEFAULT_CACHE_DIR", cache_dir)
    run_cli(["cache", "list"])
    out = capsys.readouterr().out
    assert "ds-1" in out
