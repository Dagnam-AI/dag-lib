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
        _native_train_tf: Any = None,
        _native_test_tf: Any = None,
        _native_train_flax: Any = None,
        _native_test_flax: Any = None,
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
        # Framework-native dataset objects populated by system_loader when
        # tensorflow_datasets or jax-native loaders are available (16.72-bb/16.82-bb).
        self._native_train_tf: Any = _native_train_tf
        self._native_test_tf: Any = _native_test_tf
        self._native_train_flax: Any = _native_train_flax
        self._native_test_flax: Any = _native_test_flax
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
    # Native TF / FLAX adapters (16.72-bb and 16.82-bb)
    # ------------------------------------------------------------------

    def _native_to_tensorflow(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float,
        seed: int,
        map_fn=None,
        batch_map_fn=None,
    ):
        """Convert a torchvision-style native dataset into a tf.data.Dataset.

        Materializes all samples in memory as numpy arrays then constructs a
        ``tf.data.Dataset.from_tensor_slices``. Intended for small benchmark
        datasets (MNIST, CIFAR-10, Fashion-MNIST). For larger datasets the
        caller should use ``_native_train_tf`` / ``_native_test_tf`` set by
        the TF-specific system loader (see ``_load_native_tf``).
        """
        import numpy as np
        import tensorflow as tf

        native_train = self._native_train
        native_test = self._native_test

        if isinstance(native_train, tuple):
            # numpy tuple datasets (IMDB)
            x_train, y_train = native_train
            x_test, y_test = native_test
            if x_train.dtype == object:
                x_train = self._pad_sequences(x_train)
                x_test = self._pad_sequences(x_test)
            if split == "test":
                x, y = np.asarray(x_test), np.asarray(y_test).astype(np.int64)
            else:
                n = len(x_train)
                n_val = int(n * val_ratio)
                if split == "val":
                    x = np.asarray(x_train[-n_val:]) if n_val > 0 else np.asarray([])
                    y = np.asarray(y_train[-n_val:]).astype(np.int64) if n_val > 0 else np.asarray([], dtype=np.int64)
                else:
                    x = np.asarray(x_train[:-n_val] if n_val > 0 else x_train)
                    y = np.asarray(y_train[:-n_val] if n_val > 0 else y_train).astype(np.int64)
        else:
            # torchvision-style: iterate to materialize
            import torch as _torch
            source = native_test if (split == "test" and native_test is not None) else native_train
            images = []
            labels = []
            for i in range(len(source)):
                img, lbl = source[i]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                images.append(img)
                labels.append(int(lbl))
            x = np.stack(images)
            y = np.array(labels, dtype=np.int64)
            # For split='val' or 'train' on the training set, apply val cut.
            if split in ("train", "val") and native_test is not None:
                n = len(x)
                n_val = int(n * val_ratio)
                rng = np.random.default_rng(seed)
                order = rng.permutation(n)
                val_idx = order[:n_val]
                train_idx = order[n_val:]
                if split == "val":
                    x, y = x[val_idx], y[val_idx]
                else:
                    x, y = x[train_idx], y[train_idx]

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=max(len(x), 1024), seed=seed)
        if map_fn is not None:
            ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if batch_map_fn is not None:
            ds = ds.map(batch_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _native_to_flax(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float,
        seed: int,
        transform_fn=None,
        batch_transform_fn=None,
    ) -> list:
        """Convert a torchvision-style native dataset into a list of FlaxBatch."""
        import numpy as np
        import jax.numpy as jnp
        from dagnam.data.loaders.flax_loader import FlaxBatch

        native_train = self._native_train
        native_test = self._native_test

        if isinstance(native_train, tuple):
            x_train, y_train = native_train
            x_test, y_test = native_test
            if x_train.dtype == object:
                x_train = self._pad_sequences(x_train)
                x_test = self._pad_sequences(x_test)
            if split == "test":
                x, y = np.asarray(x_test), np.asarray(y_test).astype(np.int64)
            else:
                n = len(x_train)
                n_val = int(n * val_ratio)
                if split == "val":
                    x = np.asarray(x_train[-n_val:]) if n_val > 0 else np.asarray([])
                    y = np.asarray(y_train[-n_val:]).astype(np.int64) if n_val > 0 else np.asarray([], dtype=np.int64)
                else:
                    x = np.asarray(x_train[:-n_val] if n_val > 0 else x_train)
                    y = np.asarray(y_train[:-n_val] if n_val > 0 else y_train).astype(np.int64)
        else:
            source = native_test if (split == "test" and native_test is not None) else native_train
            images = []
            labels = []
            for i in range(len(source)):
                img, lbl = source[i]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                images.append(img)
                labels.append(int(lbl))
            x = np.stack(images)
            y = np.array(labels, dtype=np.int64)
            if split in ("train", "val") and native_test is not None:
                n = len(x)
                n_val = int(n * val_ratio)
                rng_np = np.random.default_rng(seed)
                order = rng_np.permutation(n)
                if split == "val":
                    x, y = x[order[:n_val]], y[order[:n_val]]
                else:
                    x, y = x[order[n_val:]], y[order[n_val:]]

        if shuffle:
            rng_np = np.random.default_rng(seed)
            order = rng_np.permutation(len(x))
            x, y = x[order], y[order]

        batches = []
        for start in range(0, len(x), batch_size):
            batch_x = x[start : start + batch_size]
            batch_y = y[start : start + batch_size]
            if transform_fn is not None:
                batch_x = np.stack([transform_fn(s) for s in batch_x])
            feat = jnp.asarray(batch_x)
            lbl = jnp.asarray(batch_y)
            batch = FlaxBatch(features=feat, labels=lbl)
            if batch_transform_fn is not None:
                f, l = batch_transform_fn(batch.features, batch.labels)
                batch = FlaxBatch(features=f, labels=l)
            batches.append(batch)
        return batches

    def _native_tensorflow_dataset(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float = 0.1,
        seed: int = 42,
        map_fn=None,
        batch_map_fn=None,
    ):
        """Route to a TF-native dataset when ``_native_train_tf`` is set.

        Partitions the native train split into train/val subsets so that
        callers requesting ``split='val'`` get a distinct slice instead of the
        full training set.
        """
        import tensorflow as tf

        native_train_tf = getattr(self, "_native_train_tf", None)
        native_test_tf = getattr(self, "_native_test_tf", None)

        if split == "test":
            ds = native_test_tf if native_test_tf is not None else native_train_tf
        elif split == "val":
            if native_train_tf is None:
                raise ValueError("No native TF dataset available for 'val' split")
            cardinality = tf.data.experimental.cardinality(native_train_tf).numpy()
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY or cardinality < 0:
                # Fall back to materializing the count; prefer to ask tfds for it
                # via the cached InfoDatasetBuilder when possible, but for unknown
                # sources we must iterate once.
                cardinality = sum(1 for _ in native_train_tf)
            n_val = max(1, int(cardinality * val_ratio))
            ds = native_train_tf.take(n_val)
        else:  # train
            if native_train_tf is None:
                raise ValueError("No native TF dataset available for 'train' split")
            cardinality = tf.data.experimental.cardinality(native_train_tf).numpy()
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY or cardinality < 0:
                cardinality = sum(1 for _ in native_train_tf)
            n_val = max(1, int(cardinality * val_ratio))
            ds = native_train_tf.skip(n_val)

        if ds is None:
            raise ValueError(f"No native TF dataset for split '{split}'")

        if shuffle:
            ds = ds.shuffle(buffer_size=max(batch_size * 16, 1024), seed=seed)
        if map_fn is not None:
            ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if batch_map_fn is not None:
            ds = ds.map(batch_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _native_flax_dataset(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float = 0.1,
        seed: int = 42,
        transform_fn=None,
        batch_transform_fn=None,
    ) -> list:
        """Route to a FLAX-native dataset when ``_native_train_flax`` is set.

        The native FLAX path stores ``list[FlaxBatch]`` at a native batch size
        chosen by the loader. This helper flattens that list to samples, then
        applies the caller-requested split/shuffle/batch semantics plus
        optional transforms so val/test stay deterministic.
        """
        import numpy as np
        import jax.numpy as jnp
        from dagnam.data.loaders.flax_loader import FlaxBatch

        native_train_flax = getattr(self, "_native_train_flax", None)
        native_test_flax = getattr(self, "_native_test_flax", None)

        if split == "test" and native_test_flax is not None:
            source_batches = native_test_flax
        elif split in ("train", "val"):
            if native_train_flax is None:
                raise ValueError(f"No native FLAX dataset for split '{split}'")
            source_batches = native_train_flax
        else:
            source_batches = native_train_flax or []

        if not source_batches:
            return []

        # Flatten to per-sample arrays so we can re-split and rebatch.
        features_list = [np.asarray(b.features) for b in source_batches]
        labels_list = [np.asarray(b.labels) for b in source_batches]
        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        if split in ("train", "val"):
            n = len(all_features)
            n_val = max(1, int(n * val_ratio))
            rng_np = np.random.default_rng(seed)
            order = rng_np.permutation(n)
            if split == "val":
                keep = order[:n_val]
            else:
                keep = order[n_val:]
            all_features = all_features[keep]
            all_labels = all_labels[keep]

        if shuffle:
            rng_np = np.random.default_rng(seed)
            order = rng_np.permutation(len(all_features))
            all_features = all_features[order]
            all_labels = all_labels[order]

        batches: list[FlaxBatch] = []
        for start in range(0, len(all_features), batch_size):
            chunk_x = all_features[start : start + batch_size]
            chunk_y = all_labels[start : start + batch_size]
            if transform_fn is not None:
                chunk_x = np.stack([transform_fn(s) for s in chunk_x])
            feat = jnp.asarray(chunk_x)
            lbl = jnp.asarray(chunk_y)
            batch = FlaxBatch(features=feat, labels=lbl)
            if batch_transform_fn is not None:
                f, l = batch_transform_fn(batch.features, batch.labels)
                batch = FlaxBatch(features=f, labels=l)
            batches.append(batch)
        return batches

    def _try_upgrade_to_native_tf(self) -> bool:
        """Upgrade a PT-native system dataset to a TF-native dataset via tfds.

        Returns True if the upgrade succeeded and populated
        ``_native_train_tf`` / ``_native_test_tf``. False when tfds isn't
        available or the dataset doesn't map to a known tfds name; the caller
        should then fall through to ``_native_to_tensorflow`` (in-memory).
        """
        if getattr(self, "_native_train_tf", None) is not None:
            return True
        if not self._raw_meta.get("source_type") == "system":
            return False
        try:
            from dagnam.data.loaders.system_loader import (
                _resolve_tfds_name,
                resolve_system_dataset_tf,
            )
            import tensorflow_datasets  # noqa: F401
        except ImportError:
            return False

        if _resolve_tfds_name(self._raw_meta) is None:
            return False

        upgraded = resolve_system_dataset_tf(self._raw_meta)
        if upgraded._native_train_tf is None:
            return False
        # Copy upgraded native handles onto self so subsequent calls use them.
        self._native_train_tf = upgraded._native_train_tf
        self._native_test_tf = upgraded._native_test_tf
        return True

    def _try_upgrade_to_native_flax(self) -> bool:
        """Upgrade a PT-native system dataset to a FLAX-native dataset via tfds."""
        if getattr(self, "_native_train_flax", None) is not None:
            return True
        if not self._raw_meta.get("source_type") == "system":
            return False
        try:
            from dagnam.data.loaders.system_loader import (
                _resolve_tfds_name,
                resolve_system_dataset_flax,
            )
            import tensorflow_datasets  # noqa: F401
        except ImportError:
            return False

        if _resolve_tfds_name(self._raw_meta) is None:
            return False

        upgraded = resolve_system_dataset_flax(self._raw_meta)
        if upgraded._native_train_flax is None:
            return False
        self._native_train_flax = upgraded._native_train_flax
        self._native_test_flax = upgraded._native_test_flax
        return True


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

        Contract: split order is defined by Python's stdlib
        ``random.Random(seed).shuffle``. Keep all framework loaders on this
        RNG unless the split contract is intentionally versioned.
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
        column_roles: dict[str, str] | None = None,
        map_fn=None,
        batch_map_fn=None,
    ) -> "tf.data.Dataset":  # noqa: F821
        """Create a TensorFlow Dataset for the specified split.

        Supports tabular (CSV/TSV/JSON/JSONL), image-folder, and audio-folder
        datasets, plus system datasets (via the native path).

        Args:
            column_roles: Optional mapping of column names to roles for
                tabular datasets (``{"x": "feature", "label": "target"}``).
                Ignored for image/audio formats.

        Raises ImportError if tensorflow is not installed.
        Raises ValueError for unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'."
            )

        # Format validation — before TF import so unsupported formats raise
        # ValueError regardless of install state.
        fmt = self.format.lower()
        supported_formats = {
            "csv", "tsv", "json", "jsonl", "image_folder", "audio_folder",
        }
        is_system_with_native = (
            self._native_train is not None
            or getattr(self, "_native_train_tf", None) is not None
        )
        is_audio_via_type = (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        )
        if (
            fmt not in supported_formats
            and not is_system_with_native
            and not is_audio_via_type
        ):
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

        if shuffle is None:
            shuffle = split == "train"

        # --- Native dataset path (system datasets) ---
        if getattr(self, "_native_train_tf", None) is not None:
            return self._native_tensorflow_dataset(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                map_fn=map_fn,
                batch_map_fn=batch_map_fn,
            )
        if self._native_train is not None:
            # Try to upgrade to a native TF path via tensorflow_datasets if available.
            upgraded = self._try_upgrade_to_native_tf()
            if upgraded:
                return self._native_tensorflow_dataset(
                    split=split,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    val_ratio=val_ratio,
                    seed=seed,
                    map_fn=map_fn,
                    batch_map_fn=batch_map_fn,
                )
            # Legacy native path: convert PyTorch-style native datasets to tf.data.
            return self._native_to_tensorflow(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                map_fn=map_fn,
                batch_map_fn=batch_map_fn,
            )

        fmt = self.format.lower()

        # Image folder
        if fmt == "image_folder":
            from dagnam.data.loaders.image_folder_loader import (
                create_tensorflow_dataset as create_image_tf,
            )
            return create_image_tf(
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

        # Audio folder
        if fmt == "audio_folder" or (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        ):
            from dagnam.data.loaders.audio_loader import (
                create_tensorflow_dataset as create_audio_tf,
            )
            return create_audio_tf(
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

        # Tabular datasets
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(
                f"Unsupported format for TensorFlow dataset: {self.format}"
            )

        from dagnam.data.loaders.tf_loader import create_tensorflow_dataset

        return create_tensorflow_dataset(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            column_roles=column_roles,
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
        column_roles: dict[str, str] | None = None,
        transform_fn=None,
        batch_transform_fn=None,
    ) -> list:
        """Create a list of Flax batches for the specified split.

        Supports tabular (CSV/TSV/JSON/JSONL), image-folder, and audio-folder
        datasets, plus system datasets (via the native path).

        Args:
            column_roles: Optional mapping of column names to roles for
                tabular datasets. Ignored for image/audio formats.

        Raises ImportError if jax/flax is not installed.
        Raises ValueError for unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"Unknown split: {split}. Use 'train', 'val', or 'test'."
            )

        # Format validation — before JAX import.
        fmt = self.format.lower()
        supported_formats = {
            "csv", "tsv", "json", "jsonl", "image_folder", "audio_folder",
        }
        is_system_with_native = (
            self._native_train is not None
            or getattr(self, "_native_train_flax", None) is not None
        )
        is_audio_via_type = (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        )
        if (
            fmt not in supported_formats
            and not is_system_with_native
            and not is_audio_via_type
        ):
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

        if shuffle is None:
            shuffle = split == "train"

        # --- Native dataset path ---
        if getattr(self, "_native_train_flax", None) is not None:
            return self._native_flax_dataset(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                transform_fn=transform_fn,
                batch_transform_fn=batch_transform_fn,
            )
        if self._native_train is not None:
            upgraded = self._try_upgrade_to_native_flax()
            if upgraded:
                return self._native_flax_dataset(
                    split=split,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    val_ratio=val_ratio,
                    seed=seed,
                    transform_fn=transform_fn,
                    batch_transform_fn=batch_transform_fn,
                )
            return self._native_to_flax(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                transform_fn=transform_fn,
                batch_transform_fn=batch_transform_fn,
            )

        fmt = self.format.lower()

        # Image folder
        if fmt == "image_folder":
            from dagnam.data.loaders.image_folder_loader import (
                create_flax_dataset as create_image_flax,
            )
            return create_image_flax(
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

        # Audio folder
        if fmt == "audio_folder" or (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        ):
            from dagnam.data.loaders.audio_loader import (
                create_flax_dataset as create_audio_flax,
            )
            return create_audio_flax(
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

        # Tabular
        if fmt not in ("csv", "tsv", "json", "jsonl"):
            raise ValueError(
                f"Unsupported format for Flax dataset: {self.format}"
            )

        from dagnam.data.loaders.flax_loader import create_flax_dataset

        return create_flax_dataset(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            column_roles=column_roles,
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
