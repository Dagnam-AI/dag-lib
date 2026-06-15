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
    scan_class_samples,
    select_split_indices,
    split_indices,
)

# ------------------------------------------------------------------
# media_utils tests
# ------------------------------------------------------------------


class TestDiscoverClassFolders:
    """Tests for discover_class_folders utility."""

    def test_explicit_splits_with_class_folders(self, tmp_path: Path) -> None:
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

    def test_unsplit_class_folders(self, tmp_path: Path) -> None:
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

    def test_partial_splits_treated_as_unsplit(self, tmp_path: Path) -> None:
        """Only train folder present — treated as unsplit."""
        for cls in ("a", "b"):
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
            (d / "x.jpg").write_bytes(b"x")

        layout = discover_class_folders(tmp_path)
        # Only train exists, no val/test — still counts as explicit split
        assert layout.has_explicit_splits is True
        assert set(layout.splits) == {"train"}

    def test_empty_directory_returns_empty_layout(self, tmp_path: Path) -> None:
        """Empty directory returns layout with no classes."""
        layout = discover_class_folders(tmp_path)
        assert layout.has_explicit_splits is False
        assert layout.class_names == []
        assert layout.splits == []


class TestSplitIndices:
    """Tests for deterministic split_indices utility."""

    def test_basic_split(self) -> None:
        """Splits indices deterministically."""
        train, val, test = split_indices(100, val_ratio=0.1, test_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        # No overlap
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()

    def test_deterministic(self) -> None:
        """Same seed produces same split."""
        result1 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=123)
        result2 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=123)
        assert result1 == result2

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different splits."""
        result1 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=1)
        result2 = split_indices(50, val_ratio=0.2, test_ratio=0.1, seed=2)
        assert result1 != result2

    def test_zero_val_and_test(self) -> None:
        """All indices go to train when val/test ratios are 0."""
        train, val, test = split_indices(20, val_ratio=0.0, test_ratio=0.0, seed=42)
        assert len(train) == 20
        assert len(val) == 0
        assert len(test) == 0


class TestSelectSplitIndicesLazy:
    """The lazy selector must match the full partition and memoize the shuffle."""

    def test_matches_full_partition(self) -> None:
        train, val, test = split_indices(1000, val_ratio=0.1, test_ratio=0.1, seed=42)
        assert select_split_indices(1000, "train", seed=42) == train
        assert select_split_indices(1000, "val", seed=42) == val
        assert select_split_indices(1000, "test", seed=42) == test

    def test_sizes_unchanged(self) -> None:
        assert len(select_split_indices(1000, "train")) == 800
        assert len(select_split_indices(1000, "val")) == 100
        assert len(select_split_indices(1000, "test")) == 100

    def test_partition_is_disjoint_and_total(self) -> None:
        n = 777
        parts = [select_split_indices(n, s, seed=7) for s in ("train", "val", "test")]
        combined = sorted(i for part in parts for i in part)
        assert combined == list(range(n))

    def test_permutation_memoized(self) -> None:
        from dagnam.data.loaders import media

        media._shuffled_indices.cache_clear()
        select_split_indices(1000, "train", seed=42)
        select_split_indices(1000, "val", seed=42)
        select_split_indices(1000, "test", seed=42)
        info = media._shuffled_indices.cache_info()
        assert info.misses == 1
        assert info.hits == 2

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(KeyError):
            select_split_indices(10, "holdout")


class TestScanClassSamples:
    """The cached directory scan enumerates samples once per ``(root, mtime)``."""

    def _make_tree(self, root: Path) -> None:
        for cls in ("cat", "dog"):
            d = root / cls
            d.mkdir()
            (d / "a.jpg").write_bytes(b"x")
            (d / "b.png").write_bytes(b"y")
            (d / "skip.txt").write_bytes(b"z")

    def test_enumerates_samples_sorted_by_class_then_name(self, tmp_path: Path) -> None:
        self._make_tree(tmp_path)
        samples, classes = scan_class_samples(tmp_path)
        assert classes == ["cat", "dog"]
        names = [(p.name, idx) for p, idx in samples]
        assert names == [
            ("a.jpg", 0),
            ("b.png", 0),
            ("a.jpg", 1),
            ("b.png", 1),
        ]

    def test_ignores_non_image_files_and_hidden_dirs(self, tmp_path: Path) -> None:
        self._make_tree(tmp_path)
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "x.jpg").write_bytes(b"q")
        samples, classes = scan_class_samples(tmp_path)
        assert classes == ["cat", "dog"]
        assert all(p.suffix.lower() != ".txt" for p, _ in samples)

    def test_repeated_scan_uses_cache(self, tmp_path: Path) -> None:
        from dagnam.data.loaders import media

        media._scan_class_samples_cached.cache_clear()
        self._make_tree(tmp_path)
        scan_class_samples(tmp_path)
        scan_class_samples(tmp_path)
        info = media._scan_class_samples_cached.cache_info()
        assert info.misses == 1
        assert info.hits == 1

    def test_returned_lists_are_independent_copies(self, tmp_path: Path) -> None:
        self._make_tree(tmp_path)
        samples1, classes1 = scan_class_samples(tmp_path)
        samples1.append((Path("nope"), 99))
        classes1.append("ghost")
        samples2, classes2 = scan_class_samples(tmp_path)
        assert (Path("nope"), 99) not in samples2
        assert "ghost" not in classes2

    def test_cache_invalidates_when_file_added_to_existing_class(self, tmp_path: Path) -> None:
        """Adding an image to an EXISTING class dir must surface on the next scan.

        Regression: the cache key previously used only the root's whole-second
        mtime. Adding a file inside an existing class folder bumps that folder's
        mtime, not the root's, so the stale single-file list was returned. The
        key now includes each class dir's mtime, so the entry invalidates.
        """
        import os

        from dagnam.data.loaders import media

        media._scan_class_samples_cached.cache_clear()
        cls = tmp_path / "cat"
        cls.mkdir()
        (cls / "a.jpg").write_bytes(b"x")
        samples_before, _ = scan_class_samples(tmp_path)
        assert [p.name for p, _ in samples_before] == ["a.jpg"]

        # Add a file to the existing class dir, then force its mtime forward so
        # the assertion is independent of filesystem mtime resolution. Pre-fix
        # (root-only key) this stayed cached and returned only "a.jpg".
        (cls / "b.jpg").write_bytes(b"y")
        bumped = cls.stat().st_mtime_ns + 5_000_000_000
        os.utime(cls, ns=(bumped, bumped))

        samples_after, _ = scan_class_samples(tmp_path)
        assert sorted(p.name for p, _ in samples_after) == ["a.jpg", "b.jpg"]


class TestGatherImageSamplesDelegates:
    """``_gather_image_samples`` delegates to the cached ``scan_class_samples``."""

    def test_delegates_to_scan_class_samples(self, tmp_path: Path) -> None:
        from dagnam.data.loaders import image_folder

        sentinel = ([(tmp_path / "z.jpg", 0)], ["z"])
        with patch(
            "dagnam.data.loaders.image_folder.scan_class_samples",
            return_value=sentinel,
        ) as mock_scan:
            result = image_folder._gather_image_samples(tmp_path)

        assert result == sentinel
        mock_scan.assert_called_once_with(tmp_path)


class TestFolderLayout:
    """Tests for FolderLayout dataclass."""

    def test_immutable(self) -> None:
        """FolderLayout is frozen."""
        layout = FolderLayout(
            has_explicit_splits=True,
            class_names=["a", "b"],
            splits=["train", "val"],
            root=Path("/tmp"),
        )
        with pytest.raises(AttributeError):
            # Deliberately assigning to a frozen dataclass field to prove it raises.
            layout.class_names = ["c"]  # pyright: ignore[reportAttributeAccessIssue]


class TestSafeArchiveExtraction:
    """Archive extraction must not write outside the cache directory."""

    def test_zip_path_traversal_is_rejected(self, tmp_path: Path) -> None:
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../escape.txt", "owned")

        with pytest.raises(ValueError, match="Unsafe archive member"):
            ensure_extracted(tmp_path)

        assert not (tmp_path.parent / "escape.txt").exists()

    def test_zip_decompression_bomb_is_rejected(self, tmp_path: Path) -> None:
        class Archive:
            def infolist(self):
                info = zipfile.ZipInfo("huge.bin")
                info.file_size = 10 * 1024 * 1024 * 1024
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Archive is too large"):
            _safe_extract_zip(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_tar_decompression_bomb_is_rejected(self, tmp_path: Path) -> None:
        class Archive:
            def getmembers(self):
                info = tarfile.TarInfo("huge.bin")
                info.size = 10 * 1024 * 1024 * 1024
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Archive is too large"):
            _safe_extract_tar(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_zip_symlink_member_is_rejected(self, tmp_path: Path) -> None:
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

    def test_tar_special_member_is_rejected(self, tmp_path: Path) -> None:
        class Archive:
            def getmembers(self):
                info = tarfile.TarInfo("device")
                info.type = tarfile.CHRTYPE
                return [info]

            def extractall(self, destination: Path):
                raise AssertionError("unsafe archive should not be extracted")

        with pytest.raises(ValueError, match="Unsafe archive member type"):
            _safe_extract_tar(Archive(), tmp_path)  # type: ignore[arg-type]

    def test_tar_path_traversal_is_rejected(self, tmp_path: Path) -> None:
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

    def test_pytorch_extra_includes_torchvision(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        assert 'pytorch = ["torch>=2.0", "torchvision>=0.15"]' in pyproject


# ------------------------------------------------------------------
# DagnamDataset dispatch tests
# ------------------------------------------------------------------


class TestImageFolderDispatch:
    """Tests that DagnamDataset dispatches image_folder to the image loader."""

    def test_image_folder_dispatches_to_loader(self, tmp_path: Path) -> None:
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

    def test_image_folder_raises_importerror_without_torch(self, tmp_path: Path) -> None:
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
