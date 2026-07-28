"""Coverage for internal-mode and error paths in dagnam.data.load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tests.typing_helpers import PytestMonkeyPatch

from dagnam import load_dataset
from dagnam._core.client import DagnamClient
from dagnam._core.exceptions import ChecksumError
from dagnam.data.load import _load_internal, _validate_internal_dataset_id

SYS_META = {
    "id": "mnist-digits",
    "name": "MNIST",
    "format": "csv",
    "dataset_type": "image",
    "num_samples": 1,
    "num_classes": 10,
    "feature_schema": None,
    "class_names": None,
    "checksum": "placeholder",
}


class TestInternalMode:
    def test_internal_uses_sidecar(self, tmp_path: Path, monkeypatch: PytestMonkeyPatch) -> None:
        meta = {
            "id": "abc",
            "name": "demo",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 1,
            "file_path": str(tmp_path / "data.csv"),
        }
        (tmp_path / "data.csv").write_text("x\n1\n")
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "abc.meta.json").write_text(json.dumps(meta))

        monkeypatch.setenv("DAGNAM_INTERNAL", "1")
        monkeypatch.setenv("DAGNAM_META_DIR", str(meta_dir))

        ds = load_dataset("abc")
        assert ds.name == "demo"

    def test_internal_legacy_meta_json(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        storage = tmp_path / "uploads"
        ds_dir = storage / "abc"
        ds_dir.mkdir(parents=True)
        meta = {
            "id": "abc",
            "name": "legacy",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 0,
            "num_classes": 0,
        }
        (ds_dir / "meta.json").write_text(json.dumps(meta))

        monkeypatch.setenv("DAGNAM_INTERNAL", "1")
        monkeypatch.setenv("DAGNAM_META_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("DAGNAM_STORAGE_PATH", str(storage))

        ds = _load_internal("abc")
        assert ds.name == "legacy"

    def test_internal_missing_everything_raises(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAGNAM_META_DIR", str(tmp_path / "nope"))
        monkeypatch.setenv("DAGNAM_STORAGE_PATH", str(tmp_path / "also_nope"))
        with pytest.raises(FileNotFoundError, match="Sidecar metadata not found"):
            _load_internal("abc")

    def test_internal_absent_file_path_raises(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        # A non-system sidecar with no file_path key at all → the str/non-empty
        # guard is False and resolution raises a clear FileNotFoundError.
        meta = {
            "id": "abc",
            "name": "demo",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 0,
            "num_classes": 0,
        }
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "abc.meta.json").write_text(json.dumps(meta))
        monkeypatch.setenv("DAGNAM_META_DIR", str(meta_dir))
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            _load_internal("abc")

    def test_internal_missing_file_path_raises(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        meta = {
            "id": "abc",
            "name": "demo",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 0,
            "num_classes": 0,
            "file_path": str(tmp_path / "missing.csv"),
        }
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "abc.meta.json").write_text(json.dumps(meta))
        monkeypatch.setenv("DAGNAM_META_DIR", str(meta_dir))
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            _load_internal("abc")

    def test_internal_system_source_resolves_native(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        meta = {
            "id": "mnist-digits",
            "name": "MNIST",
            "format": "csv",
            "dataset_type": "image",
            "num_samples": 1,
            "num_classes": 10,
            "source_type": "system",
        }
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "mnist-digits.meta.json").write_text(json.dumps(meta))
        monkeypatch.setenv("DAGNAM_META_DIR", str(meta_dir))

        native = MagicMock()
        with patch("dagnam.data.loaders.system.load_system_dataset", return_value=native):
            assert _load_internal("mnist-digits") is native

    def test_internal_system_resolve_propagates_error(
        self, tmp_path: Path, monkeypatch: PytestMonkeyPatch
    ) -> None:
        # A system dataset has no real on-disk file (the relative file_path is a
        # codegen placeholder), so a native-loader failure must propagate its
        # real cause instead of being masked as a FileNotFoundError. This is the
        # Observability fix.
        meta = {
            "id": "mnist-digits",
            "name": "MNIST",
            "format": "csv",
            "dataset_type": "image",
            "num_samples": 1,
            "num_classes": 10,
            "source_type": "system",
        }
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "mnist-digits.meta.json").write_text(json.dumps(meta))
        monkeypatch.setenv("DAGNAM_META_DIR", str(meta_dir))

        with patch(
            "dagnam.data.loaders.system.load_system_dataset",
            side_effect=RuntimeError("real loader error"),
        ):
            with pytest.raises(RuntimeError, match="real loader error"):
                _load_internal("mnist-digits")


class TestValidateInternalId:
    @pytest.mark.parametrize(
        "bad",
        ["", ".", "..", "a/b", "a\\b", "C:foo", "/abs/path"],
    )
    def test_unsafe_ids_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Unsafe dataset_id"):
            _validate_internal_dataset_id(bad)


class TestChecksumMismatch:
    def test_user_dataset_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        # Stage a file whose sha256 does NOT match the meta checksum.
        bad_content = b"corrupted"
        wrong_checksum = hashlib.sha256(b"original").hexdigest()
        meta = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "x",
            "format": "csv",
            "dataset_type": "tabular",
            "num_samples": 1,
            "num_classes": 1,
            "feature_schema": None,
            "class_names": None,
            "checksum": f"sha256:{wrong_checksum}",
        }
        dataset_id = str(meta["id"])

        def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
            # The locked flow downloads into a staging dir; write the (bad) bytes there.
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            staged = out / "data.csv"
            staged.write_bytes(bad_content)
            return staged

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(DagnamClient, "get_dataset_meta", return_value=meta),
            patch.object(DagnamClient, "download_dataset", side_effect=_fake_download),
        ):
            with pytest.raises(ChecksumError, match="Checksum mismatch"):
                load_dataset(dataset_id, cache_dir=str(tmp_path))


class TestSystemDownloadPath:
    def test_system_fallback_downloads_when_resolve_fails(self, tmp_path: Path) -> None:
        # source_type=system but resolve_system_dataset raises → fallback to
        # download_system_dataset path (covers download_system_dataset branch).
        csv_content = b"a,b\n1,2\n"
        checksum = hashlib.sha256(csv_content).hexdigest()
        meta = {
            **SYS_META,
            "checksum": f"sha256:{checksum}",
            "source_type": "system",
        }

        def _fake_download(_ds_id: str, output_dir: Path, **_kwargs: object) -> Path:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            staged = out / "data.csv"
            staged.write_bytes(csv_content)
            return staged

        with (
            patch("dagnam.data.load.get_api_key", return_value="key"),
            patch("dagnam.data.load.get_api_url", return_value="http://localhost"),
            patch.object(DagnamClient, "get_system_dataset_meta", return_value=meta),
            patch.object(
                DagnamClient, "download_system_dataset", side_effect=_fake_download
            ) as mock_dl,
            patch(
                "dagnam.data.loaders.system.load_system_dataset",
                side_effect=RuntimeError("no tfds"),
            ),
        ):
            ds = load_dataset("mnist-digits", cache_dir=str(tmp_path))
            mock_dl.assert_called_once()
            assert ds.name == "MNIST"
