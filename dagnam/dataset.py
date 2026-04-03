"""DagnamDataset class for lazy-loading and converting datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DagnamDataset:
    """Represents a loaded dataset with metadata and conversion methods.

    Data is loaded lazily — file parsing is deferred until ``to_pandas()``
    or ``to_pytorch_loader()`` is called.
    """

    def __init__(self, meta: dict, data_dir: Path) -> None:
        self.id: str = meta["id"]
        self.name: str = meta["name"]
        self.format: str = meta["format"]
        self.dataset_type: str = meta["dataset_type"]
        self.num_samples: int = meta["num_samples"]
        self.num_classes: int = meta["num_classes"]
        self.feature_schema: dict | None = meta.get("feature_schema")
        self.class_names: list[str] | None = meta.get("class_names")
        self._data_dir: Path = data_dir
        self._data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict:
        """Return a summary dictionary with the 8 required keys."""
        return {
            "id": self.id,
            "name": self.name,
            "format": self.format,
            "type": self.dataset_type,
            "samples": self.num_samples,
            "classes": self.num_classes,
            "class_names": self.class_names,
            "schema": self.feature_schema,
        }

    # ------------------------------------------------------------------
    # Pandas conversion
    # ------------------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """Load the dataset as a pandas DataFrame.

        The result is cached after the first call.  Supports CSV, TSV,
        JSON, and JSONL formats.

        Raises:
            ValueError: If the format is not supported.
            FileNotFoundError: If no matching data file exists in the cache.
        """
        if self._data is not None:
            return self._data

        fmt = self.format.lower()
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(f"Cannot load format '{self.format}' as DataFrame")

        data_file = self._find_data_file()

        if fmt == "csv":
            self._data = pd.read_csv(data_file)
        elif fmt == "tsv":
            self._data = pd.read_csv(data_file, sep="\t")
        elif fmt == "json":
            self._data = pd.read_json(data_file)
        elif fmt == "jsonl":
            self._data = pd.read_json(data_file, lines=True)

        return self._data

    # ------------------------------------------------------------------
    # PyTorch conversion
    # ------------------------------------------------------------------

    def to_pytorch_loader(
        self,
        split: str = "train",
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> "torch.utils.data.DataLoader":  # noqa: F821
        """Create a PyTorch DataLoader for the specified split.

        Dispatches to a format-specific loader (csv_loader or json_loader).
        ``shuffle`` defaults to ``True`` for train, ``False`` for val/test.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: For unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'."
            )

        fmt = self.format.lower()
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(
                f"Unsupported format for PyTorch loader: {self.format}"
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch is required for to_pytorch_loader(). "
                "Install with: uv pip install dagnam[pytorch]"
            )

        if fmt in ("csv", "tsv"):
            from dagnam.loaders.csv_loader import create_pytorch_loader
        else:
            from dagnam.loaders.json_loader import create_pytorch_loader

        if shuffle is None:
            shuffle = split == "train"

        return create_pytorch_loader(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # TensorFlow conversion
    # ------------------------------------------------------------------

    def to_tensorflow_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> "tf.data.Dataset":  # noqa: F821
        """Create a TensorFlow Dataset for the specified split.

        Raises ImportError if tensorflow is not installed.
        Raises ValueError for unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'."
            )

        fmt = self.format.lower()
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(
                f"Unsupported format for TensorFlow dataset: {self.format}"
            )

        try:
            import tensorflow  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorFlow is required for to_tensorflow_dataset(). "
                "Install with: uv pip install dagnam[tensorflow]"
            )

        from dagnam.loaders.tf_loader import create_tensorflow_dataset

        if shuffle is None:
            shuffle = split == "train"

        return create_tensorflow_dataset(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Flax/JAX conversion
    # ------------------------------------------------------------------

    def to_flax_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> list:
        """Create a list of Flax batches for the specified split.

        Raises ImportError if jax/flax is not installed.
        Raises ValueError for unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'."
            )

        fmt = self.format.lower()
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(
                f"Unsupported format for Flax dataset: {self.format}"
            )

        try:
            import jax  # noqa: F401
        except ImportError:
            raise ImportError(
                "JAX is required for to_flax_dataset(). "
                "Install with: uv pip install dagnam[flax]"
            )

        from dagnam.loaders.flax_loader import create_flax_dataset

        if shuffle is None:
            shuffle = split == "train"

        return create_flax_dataset(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_data_file(self) -> Path:
        """Locate the data file in ``_data_dir`` by glob pattern.

        Excludes ``meta.json`` from JSON matches.

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        pattern_map: dict[str, list[str]] = {
            "csv": ["*.csv"],
            "tsv": ["*.tsv"],
            "json": ["*.json"],
            "jsonl": ["*.jsonl"],
        }

        fmt = self.format.lower()
        patterns = pattern_map.get(fmt, [f"*.{fmt}"])

        for pattern in patterns:
            for match in self._data_dir.glob(pattern):
                # Exclude meta.json from json matches
                if fmt == "json" and match.name == "meta.json":
                    continue
                return match

        raise FileNotFoundError(
            f"No data file matching {patterns} in {self._data_dir}"
        )
