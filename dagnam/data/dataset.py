"""DagnamDataset class for lazy-loading and converting datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd


def _wrap_collate(collate_fn=None, batch_transform=None):
    """Apply an optional batch transform after PyTorch collation."""
    if batch_transform is None:
        return collate_fn

    def wrapped(batch):
        if collate_fn is None:
            from torch.utils.data._utils.collate import default_collate

            collated = default_collate(batch)
        else:
            collated = collate_fn(batch)
        return batch_transform(collated)

    return wrapped


def _with_collate(loader, collate_fn=None, batch_transform=None):
    """Rebuild a DataLoader with hook-aware collation when needed.

    Preserves the original loader's sampler so that shuffle behavior is
    retained (PyTorch's DataLoader uses RandomSampler internally when
    shuffle=True, so handing the sampler over preserves that behavior).
    """
    wrapped_collate = _wrap_collate(collate_fn, batch_transform)
    if wrapped_collate is None:
        return loader

    from torch.utils.data import DataLoader

    # When a sampler is provided, PyTorch requires shuffle to be False/None.
    # The sampler itself encodes the shuffle behavior of the original loader.
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        collate_fn=wrapped_collate,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )


class _TransformDataset:
    """Map-style dataset wrapper that applies sample and target hooks."""

    def __init__(self, dataset, transform=None, target_transform=None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, tuple) and len(item) >= 2:
            data = item[0]
            target = item[1]
            rest = item[2:]
            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if rest:
                return (data, target, *rest)
            return data, target
        if self.transform is not None:
            return self.transform(item)
        return item


class DagnamDataset:
    """Represents a loaded dataset with metadata and conversion methods.

    Data is loaded lazily — file parsing is deferred until ``to_pandas()``
    or ``to_pytorch_loader()`` is called.

    For system datasets loaded via native libraries (torchvision, etc.),
    ``_native_train`` and ``_native_test`` are populated directly and
    ``to_pytorch_loader()`` uses them instead of loading from a file.
    """

    def __init__(
        self,
        meta: dict,
        data_dir: Path,
        _native_train: Any = None,
        _native_test: Any = None,
    ) -> None:
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
        self._native_train: Any = _native_train
        self._native_test: Any = _native_test
        self._raw_meta: dict = meta

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
        column_roles: dict[str, str] | None = None,
        transform=None,
        target_transform=None,
        collate_fn=None,
        batch_transform=None,
        waveform_transform=None,
        spectrogram_transform=None,
    ) -> "torch.utils.data.DataLoader":  # noqa: F821
        """Create a PyTorch DataLoader for the specified split.

        When ``_native_train`` / ``_native_test`` are set (system datasets),
        uses the native dataset directly.  Otherwise dispatches to a
        format-specific loader (csv_loader or json_loader).

        ``shuffle`` defaults to ``True`` for train, ``False`` for val/test.

        Args:
            column_roles: Optional mapping of column names to roles
                (e.g. ``{"x": "feature", "label": "target"}``).
                Only used by tabular loaders (CSV/JSON). Ignored by
                image and audio loaders.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: For unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'."
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch is required for to_pytorch_loader(). "
                "Install with: uv pip install dagnam[pytorch]"
            )

        if shuffle is None:
            shuffle = split == "train"

        # --- Native dataset path (system datasets via torchvision etc.) ---
        if self._native_train is not None:
            return self._native_pytorch_loader(
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                transform=transform,
                target_transform=target_transform,
                collate_fn=_wrap_collate(collate_fn, batch_transform),
            )

        # --- File-based path (user datasets) ---
        fmt = self.format.lower()

        # Image folder datasets
        if fmt == "image_folder":
            from dagnam.data.loaders.image_folder_loader import (
                create_pytorch_loader as create_image_loader,
            )

            return create_image_loader(
                dagnam_ds=self,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                transform=transform,
                target_transform=target_transform,
                collate_fn=_wrap_collate(collate_fn, batch_transform),
            )

        # Audio folder datasets
        if fmt == "audio_folder" or (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        ):
            from dagnam.data.loaders.audio_loader import (
                create_pytorch_loader as create_audio_loader,
            )

            return create_audio_loader(
                dagnam_ds=self,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                waveform_transform=waveform_transform,
                spectrogram_transform=spectrogram_transform,
                target_transform=target_transform,
                collate_fn=_wrap_collate(collate_fn, batch_transform),
            )

        # Tabular datasets (CSV, TSV, JSON, JSONL)
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(
                f"Unsupported format for PyTorch loader: {self.format}"
            )

        if fmt in ("csv", "tsv"):
            from dagnam.data.loaders.csv_loader import create_pytorch_loader
        else:
            from dagnam.data.loaders.json_loader import create_pytorch_loader

        loader = create_pytorch_loader(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            column_roles=column_roles,
        )
        return _with_collate(loader, collate_fn, batch_transform)

    def _native_pytorch_loader(
        self,
        split: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        val_ratio: float,
        seed: int,
        transform=None,
        target_transform=None,
        collate_fn=None,
    ) -> "torch.utils.data.DataLoader":  # noqa: F821
        """Build a DataLoader from native train/test datasets."""
        import torch
        from torch.utils.data import DataLoader, random_split

        native_train = self._native_train
        native_test = self._native_test

        # Handle IMDB-style tuple datasets (numpy arrays)
        if isinstance(native_train, tuple):
            return self._native_numpy_loader(
                split,
                batch_size,
                num_workers,
                shuffle,
                val_ratio,
                seed,
                transform=transform,
                target_transform=target_transform,
                collate_fn=collate_fn,
            )

        # Handle torchvision map-style datasets
        if split == "test":
            ds = native_test if native_test is not None else native_train
        elif split == "val":
            n_val = int(len(native_train) * val_ratio)
            n_train = len(native_train) - n_val
            _, ds = random_split(
                native_train, [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
            )
        else:  # train
            n_val = int(len(native_train) * val_ratio)
            n_train = len(native_train) - n_val
            ds, _ = random_split(
                native_train, [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
            )

        if transform is not None or target_transform is not None:
            ds = _TransformDataset(ds, transform, target_transform)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            collate_fn=collate_fn,
        )

    def _native_numpy_loader(
        self,
        split: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        val_ratio: float,
        seed: int,
        transform=None,
        target_transform=None,
        collate_fn=None,
    ) -> "torch.utils.data.DataLoader":  # noqa: F821
        """Build a DataLoader from numpy array tuples (e.g. IMDB)."""
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x_train, y_train = self._native_train
        x_test, y_test = self._native_test

        if split == "test":
            # IMDB sequences are variable-length object arrays — pad them
            if x_test.dtype == object:
                x_test = self._pad_sequences(x_test)
            x_t = torch.from_numpy(np.asarray(x_test)).long()
            y_t = torch.from_numpy(np.asarray(y_test)).float().unsqueeze(1)
            ds = TensorDataset(x_t, y_t)
        else:
            if x_train.dtype == object:
                x_train = self._pad_sequences(x_train)
            n_val = int(len(x_train) * val_ratio)
            if split == "val":
                x = torch.from_numpy(np.asarray(x_train[-n_val:])).long()
                y = torch.from_numpy(np.asarray(y_train[-n_val:])).float().unsqueeze(1)
            else:
                x = torch.from_numpy(np.asarray(x_train[:-n_val] if n_val > 0 else x_train)).long()
                y = torch.from_numpy(np.asarray(y_train[:-n_val] if n_val > 0 else y_train)).float().unsqueeze(1)
            ds = TensorDataset(x, y)

        if transform is not None or target_transform is not None:
            ds = _TransformDataset(ds, transform, target_transform)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            collate_fn=collate_fn,
        )

    @staticmethod
    def _pad_sequences(sequences, maxlen: int = 200, num_words: int = 20000):
        """Pad/truncate variable-length integer sequences (e.g. IMDB)."""
        import numpy as np
        result = np.zeros((len(sequences), maxlen), dtype=np.int32)
        for i, seq in enumerate(sequences):
            filtered = [w if w < num_words else 0 for w in seq]
            trunc = filtered[:maxlen]
            result[i, : len(trunc)] = trunc
        return result

    # ------------------------------------------------------------------
    # Raw sample/array access
    # ------------------------------------------------------------------

    def iter_samples(
        self,
        split: str = "train",
        decoded: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Yield raw samples for a split when data is already available."""
        del decoded  # Reserved for future media decoding support.

        if isinstance(self._data, dict) and split in self._data:
            yield from self._data[split]
            return

        if isinstance(self._data, list):
            yield from self._data
            return

        native = self._native_test if split == "test" else self._native_train
        if isinstance(native, tuple) and len(native) == 2:
            features, labels = native
            for feature, label in zip(features, labels):
                yield feature, label
            return

        if native is not None:
            for index in range(len(native)):
                yield native[index]
            return

        if self.format.lower() in ("csv", "tsv", "json", "jsonl"):
            yield from self._iter_tabular_file_samples(
                split=split,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            return

        raise ValueError(
            f"Raw sample iteration is not available for format '{self.format}'"
        )

    def to_arrays(
        self,
        split: str = "train",
        decoded: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Return features and labels as NumPy arrays for generated pipelines."""
        import numpy as np

        features = []
        labels = []
        for item in self.iter_samples(
            split=split,
            decoded=decoded,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        ):
            if isinstance(item, tuple) and len(item) == 2:
                feature, label = item
            else:
                feature, label = item, None
            features.append(feature)
            labels.append(label)

        if labels and all(label is not None for label in labels):
            return np.asarray(features), np.asarray(labels)
        return np.asarray(features), None

    def _iter_tabular_file_samples(
        self,
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ):
        """Yield numeric feature rows and encoded labels from a tabular file."""
        df = self.to_pandas()
        label_col = self._detect_label_column(df)
        labels = self._encode_label_values(df[label_col])
        feature_cols = [col for col in df.columns if col != label_col]
        features = df[feature_cols].select_dtypes(include="number").values

        indices = self._split_indices(
            len(df),
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        for index in indices:
            yield features[index].tolist(), labels[index]

    def _detect_label_column(self, df: pd.DataFrame) -> str:
        if self.feature_schema and "columns" in self.feature_schema:
            for col_info in self.feature_schema["columns"]:
                if col_info.get("type") == "categorical":
                    return col_info["name"]
        return df.columns[-1]

    def _encode_label_values(self, series: pd.Series) -> list:
        if self.class_names:
            mapping = {name: idx for idx, name in enumerate(self.class_names)}
            return series.map(mapping).tolist()
        encoded, _ = pd.factorize(series)
        return encoded.tolist()

    @staticmethod
    def _split_indices(
        n: int,
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> list[int]:
        """Compute deterministic index ranges for train/val/test splits.

        Uses the same shuffle behavior as ``csv_loader`` / ``json_loader``
        so that ``to_arrays()`` and ``to_pytorch_loader()`` produce identical
        splits with the same seed.
        """
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        indices = list(range(n))
        # Always shuffle for determinism parity with file-based loaders,
        # which shuffle unconditionally regardless of val/test ratios.
        random.Random(seed).shuffle(indices)

        split_map = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }
        return split_map[split]

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
        map_fn=None,
        batch_map_fn=None,
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

        from dagnam.data.loaders.tf_loader import create_tensorflow_dataset

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
            map_fn=map_fn,
            batch_map_fn=batch_map_fn,
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
        transform_fn=None,
        batch_transform_fn=None,
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

        from dagnam.data.loaders.flax_loader import create_flax_dataset

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
            transform_fn=transform_fn,
            batch_transform_fn=batch_transform_fn,
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
