"""Flax/JAX conversion for DagnamDataset."""

from __future__ import annotations


class FlaxDatasetMixin:
    """Flax conversion methods."""

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
        import jax.numpy as jnp
        import numpy as np

        from dagnam.data.loaders.flax import FlaxBatch

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
                    y = (
                        np.asarray(y_train[-n_val:]).astype(np.int64)
                        if n_val > 0
                        else np.asarray([], dtype=np.int64)
                    )
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
        import jax.numpy as jnp
        import numpy as np

        from dagnam.data.loaders.flax import FlaxBatch

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

    def _try_upgrade_to_native_flax(self) -> bool:
        """Upgrade a PT-native system dataset to a FLAX-native dataset via tfds."""
        if getattr(self, "_native_train_flax", None) is not None:
            return True
        if not self._raw_meta.get("source_type") == "system":
            return False
        try:
            import tensorflow_datasets  # noqa: F401

            from dagnam.data.loaders.system import (
                _resolve_tfds_name,
                resolve_system_dataset_flax,
            )
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
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        # Format validation — before JAX import.
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
            self._native_train is not None or getattr(self, "_native_train_flax", None) is not None
        )
        is_audio_via_type = (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        )
        if fmt not in supported_formats and not is_system_with_native and not is_audio_via_type:
            raise ValueError(f"Unsupported format for Flax dataset: {self.format}")

        try:
            import jax  # noqa: F401
        except ImportError:
            raise ImportError(
                "JAX is required for to_flax_dataset(). Install with: uv pip install dagnam[flax]"
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
            from dagnam.data.loaders.image_folder import (
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
            from dagnam.data.loaders.audio import (
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
            raise ValueError(f"Unsupported format for Flax dataset: {self.format}")

        from dagnam.data.loaders.flax import create_flax_dataset

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
