"""Tests for image folder dataset loader."""

import io
from pathlib import Path
import stat
import tarfile
from unittest.mock import patch
import zipfile

import pytest

from dagnam.data.dataset import DagnamDataset
from dagnam.data.loaders.media import (
    FolderLayout,
    _safe_extract_tar,
    _safe_extract_zip,
    discover_class_folders,
    ensure_extracted,
    split_indices,
)

# ------------------------------------------------------------------
# media_utils tests
# ------------------------------------------------------------------


class TestDiscoverClassFolders:
    """Tests for discover_class_folders utility."""

    def test_explicit_splits_with_class_folders(self, tmp_path: Path):
        """Discovers train/val/test splits with class subdirectories."""
        for split in ("train", "val", "test"):
            for cls in ("cat", "dog"):
                d = tmp_path / split / cls
                d.mkdir(parents=True)
                (d / "a.jpg").write_bytes(b"x")

        layout = discover_class_folders(tmp_path)
        assert layout.has_explicit_splits is True
        assert layout.class_names == ["cat", "dog"]
        assert set(layout.splits) == {"train", "val", "test"}

    def test_unsplit_class_folders(self, tmp_path: Path):
        """Discovers class folders without explicit splits."""
        for cls in ("cat", "dog", "bird"):
            d = tmp_path / cls
            d.mkdir()
            (d / "img1.jpg").write_bytes(b"x")
            (d / "img2.png").write_bytes(b"y")

        layout = discover_class_folders(tmp_path)
        assert layout.has_explicit_splits is False
        assert layout.class_names == ["bird", "cat", "dog"]
        assert layout.splits == []

    def test_partial_splits_treated_as_unsplit(self, tmp_path: Path):
        """Only train folder present — treated as unsplit."""
        for cls in ("a", "b"):
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
            (d / "x.jpg").write_bytes(b"x")

        layout = discover_class_folders(tmp_path)
        # Only train exists, no val/test — still counts as explicit split
        assert layout.has_explicit_splits is True
        assert set(layout.splits) == {"train"}

    def test_empty_directory_returns_empty_layout(self, tmp_path: Path):
        """Empty directory returns layout with no classes."""
        layout = discover_class_folders(tmp_path)
        assert layout.has_explicit_splits is False
        assert layout.class_names == []
        assert layout.splits == []


class TestSplitIndices:
    """Tests for deterministic split_indices utility."""

    def test_basic_split(self):
        """Splits indices deterministically."""
        train, val, test = split_indices(100, val_ratio=0.1, test_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        # No overlap
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()

    def test_deterministic(self):
        """Same seed produces same split."""
        result1 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=123)
        result2 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=123)
        assert result1 == result2

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        result1 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=1)
        result2 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=2)
        assert result1 != result2

    def test_zero_val_and_test(self):
        """All indices go to train when val/test ratios are 0."""
        train, val, test = split_indices(20, val_ratio=0.0, test_ratio=0.0, seed=42)
        assert len(train) == 20
        assert len(val) == 0
        assert len(test) == 0


class TestFolderLayout:
    """Tests for FolderLayout dataclass."""

    def test_immutable(self):
        """FolderLayout is frozen."""
        layout = FolderLayout(
            has_explicit_splits=True,
            class_names=["a", "b"],
            splits=["train", "val"],
            root=Path("/tmp"),
        )
        with pytest.raises(AttributeError):
            layout.class_names = ["c"]


class TestSafeArchiveExtraction:
    """Archive extraction must not write outside the cache directory."""

    def test_zip_path_traversal_is_rejected(self, tmp_path: Path):
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../escape.txt", "owned")

        with pytest.raises(ValueError, match="Unsafe archive member"):
            ensure_extracted(tmp_path)

        assert not (tmp_path.parent / "escape.txt").exists()

    def test_zip_decompression_bomb_is_rejected(self, tmp_path: Path):
        class Archive:
            def infolist(self):
                info = zipfile.ZipInfo("huge.bin")
                info.file_size = 10 * 1024 * 1024 * 1024
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Archive is too large"):
            _safe_extract_zip(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_tar_decompression_bomb_is_rejected(self, tmp_path: Path):
        class Archive:
            def getmembers(self):
                info = tarfile.TarInfo("huge.bin")
                info.size = 10 * 1024 * 1024 * 1024
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Archive is too large"):
            _safe_extract_tar(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_zip_symlink_member_is_rejected(self, tmp_path: Path):
        class Archive:
            def infolist(self):
                info = zipfile.ZipInfo("link")
                info.file_size = 0
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Unsafe archive member link"):
            _safe_extract_zip(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_tar_special_member_is_rejected(self, tmp_path: Path):
        class Archive:
            def getmembers(self):
                info = tarfile.TarInfo("device")
                info.type = tarfile.CHRTYPE
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Unsafe archive member type"):
            _safe_extract_tar(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_tar_path_traversal_is_rejected(self, tmp_path: Path):
        archive = tmp_path / "bad.tar"
        payload = b"owned"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)

        with tarfile.open(archive, "w") as tf:
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(ValueError, match="Unsafe archive member"):
            ensure_extracted(tmp_path)

        assert not (tmp_path.parent / "escape.txt").exists()


class TestPytorchExtra:
    """Packaging metadata must install image loader dependencies."""

    def test_pytorch_extra_includes_torchvision(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        assert 'pytorch = ["torch>=2.0", "torchvision>=0.15"]' in pyproject


# ------------------------------------------------------------------
# DagnamDataset dispatch tests
# ------------------------------------------------------------------


class TestImageFolderDispatch:
    """Tests that DagnamDataset dispatches image_folder to the image loader."""

    def test_image_folder_dispatches_to_loader(self, tmp_path: Path):
        """image_folder format routes to image_folder_loader.create_pytorch_loader."""
        ds = DagnamDataset(
            {
                "id": "img-1",
                "name": "Images",
                "format": "image_folder",
                "dataset_type": "image",
                "num_samples": 4,
                "num_classes": 2,
                "class_names": ["cat", "dog"],
                "feature_schema": None,
            },
            tmp_path,
        )
        with patch(
            "dagnam.data.loaders.image_folder.create_pytorch_loader",
            return_value="loader",
        ) as mock_create:
            result = ds.to_pytorch_loader(split="train", batch_size=2, num_workers=0)

        assert result == "loader"
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["split"] == "train"
        assert call_kwargs["batch_size"] == 2
        assert call_kwargs["num_workers"] == 0

    def test_image_folder_raises_import_error_without_torch(self, tmp_path: Path):
        """Raises ImportError when torch is not available."""
        ds = DagnamDataset(
            {
                "id": "img-2",
                "name": "Images",
                "format": "image_folder",
                "dataset_type": "image",
                "num_samples": 4,
                "num_classes": 2,
            },
            tmp_path,
        )
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(ImportError, match="PyTorch"):
                ds.to_pytorch_loader(split="train", batch_size=2, num_workers=0)
