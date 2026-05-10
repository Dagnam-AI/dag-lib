"""TensorFlow conversion for DagnamDataset."""

from __future__ import annotations


class TensorflowDatasetMixin:
    """Tensorflow conversion methods."""

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
                    y = (
                        np.asarray(y_train[-n_val:]).astype(np.int64)
                        if n_val > 0
                        else np.asarray([], dtype=np.int64)
                    )
                else:
                    x = np.asarray(x_train[:-n_val] if n_val > 0 else x_train)
                    y = np.asarray(y_train[:-n_val] if n_val > 0 else y_train).astype(np.int64)
        else:
            # torchvision-style: iterate to materialize

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
            import tensorflow_datasets  # noqa: F401

            from dagnam.data.loaders.system import (
                _resolve_tfds_name,
                resolve_system_dataset_tf,
            )
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
    ) -> tf.data.Dataset:  # noqa: F821
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
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        # Format validation — before TF import so unsupported formats raise
        # ValueError regardless of install state.
        fmt = self.format.lower()
        supported_formats = {
            "csv",
            "tsv",
            "json",
            "jsonl",
            "image_folder",
            "audio_folder",
        }
        is_system_with_native = (
            self._native_train is not None or getattr(self, "_native_train_tf", None) is not None
        )
        is_audio_via_type = (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        )
        if fmt not in supported_formats and not is_system_with_native and not is_audio_via_type:
            raise ValueError(f"Unsupported format for TensorFlow dataset: {self.format}")

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
            from dagnam.data.loaders.image_folder import (
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
            from dagnam.data.loaders.audio import (
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
            raise ValueError(f"Unsupported format for TensorFlow dataset: {self.format}")

        from dagnam.data.loaders.tf import create_tensorflow_dataset

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
