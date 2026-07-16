"""Flax/JAX conversion for DagnamDataset."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from dagnam._types import IndexedDataset, SupportsNumpy
from dagnam.data.dataset._typing import DatasetMixinBase

if TYPE_CHECKING:
    import jax

    from dagnam.data.loaders.flax import FlaxBatch

ArrayTransform = Callable[[npt.ArrayLike], npt.ArrayLike]
JaxArrayFactory = Callable[[npt.ArrayLike], "jax.Array"]
BatchTransform = Callable[["jax.Array", "jax.Array"], tuple["jax.Array", "jax.Array"]]


def _column_roles_from_binding(binding: dict[str, Any]) -> dict[str, str] | None:
    roles: dict[str, str] = {}
    input_column = binding.get("input_column")
    target_column = binding.get("target_column")
    if isinstance(input_column, str) and input_column:
        roles[input_column] = "feature"
    if isinstance(target_column, str) and target_column:
        roles[target_column] = "target"
    return roles or None


class _LazyFlaxBatchStream:
    """A ``len``/index/iterate view that builds each FlaxBatch on demand.

    Backs the native (torchvision-style) flax path so a large split is never
    fully materialized into a list of batches at once — the Speech Commands
    audio OOM (~85k decoded waveforms, several GB). ``__getitem__`` decodes only
    the requested batch, so the smoke check (one batch) and a training epoch
    (one batch resident at a time) stay memory-bounded. Re-iterable, unlike a
    bare generator, so multi-epoch training works.
    """

    def __init__(self, build_batch: Callable[[int], FlaxBatch], n_batches: int) -> None:
        self._build_batch = build_batch
        self._n_batches = n_batches

    def __len__(self) -> int:
        return self._n_batches

    def __getitem__(self, index: int) -> FlaxBatch:
        if index < 0:
            index += self._n_batches
        if not 0 <= index < self._n_batches:
            raise IndexError("flax batch index out of range")
        return self._build_batch(index)

    def __iter__(self) -> Iterator[FlaxBatch]:
        for i in range(self._n_batches):
            yield self._build_batch(i)


class FlaxDatasetMixin(DatasetMixinBase):
    """Flax conversion methods."""

    def _native_to_flax(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float,
        seed: int,
        transform_fn: ArrayTransform | None = None,
        batch_transform_fn: BatchTransform | None = None,
        vocab_size: int | None = None,
        sequence_length: int | None = None,
    ) -> list[FlaxBatch] | _LazyFlaxBatchStream:
        """Convert a torchvision-style native dataset into a sequence of FlaxBatch.

        Returns a list for tuple (numpy) splits and a lazy, re-iterable
        ``_LazyFlaxBatchStream`` for indexable (torchvision-style) splits so a
        large native dataset is never fully materialized.
        """
        import jax.numpy as jnp

        from dagnam.data.loaders.flax import FlaxBatch

        as_jax_array = cast("JaxArrayFactory", jnp.asarray)

        native_train = self._native_train
        native_test = self._native_test
        if native_train is None:
            raise ValueError("No native dataset is available")

        if isinstance(native_train, tuple):
            x_train, y_train = native_train
            if native_test is not None and isinstance(native_test, tuple):
                x_test, y_test = native_test
            else:
                x_test, y_test = (), ()
            num_words = vocab_size if vocab_size is not None else 20000
            if np.asarray(x_train).dtype == object:
                # Ragged (variable-length) sequences arrive as object arrays; pad them.
                # Rectangular numeric arrays keep their natural dtype and are left as-is.
                # The embedding-derived sequence_length (G079) sets the fixed length;
                # fall back to _pad_sequences' default when unset.
                pad_kwargs = {} if sequence_length is None else {"maxlen": sequence_length}
                x_train = self._pad_sequences(
                    cast("Sequence[Sequence[int]]", x_train), num_words=num_words, **pad_kwargs
                )
                x_test = self._pad_sequences(
                    cast("Sequence[Sequence[int]]", x_test), num_words=num_words, **pad_kwargs
                )
            if split == "test":
                x = cast("npt.NDArray[np.object_]", np.asarray(x_test))
                y = np.asarray(y_test).astype(np.int64)
            else:
                n = len(x_train)
                n_val = int(n * val_ratio)
                if split == "val":
                    x = cast(
                        "npt.NDArray[np.object_]",
                        np.asarray(x_train[-n_val:]) if n_val > 0 else np.asarray([]),
                    )
                    y = (
                        np.asarray(y_train[-n_val:]).astype(np.int64)
                        if n_val > 0
                        else np.asarray([], dtype=np.int64)
                    )
                else:
                    x = cast(
                        "npt.NDArray[np.object_]",
                        np.asarray(x_train[:-n_val] if n_val > 0 else x_train),
                    )
                    y = np.asarray(y_train[:-n_val] if n_val > 0 else y_train).astype(np.int64)
        else:
            # Lazily build batches by indexing the source per batch — NEVER
            # materialize the whole split (the Speech Commands audio OOM: ~85k
            # decoded waveforms, several GB). Resolve split indices from the
            # dataset length and shuffle the cheap index array; the per-batch
            # builder decodes only ``batch_size`` samples on access. Re-iterable,
            # so multi-epoch training works.
            source = native_test if (split == "test" and native_test is not None) else native_train
            source_dataset = cast("IndexedDataset", source)
            total = len(source_dataset)
            if split in ("train", "val") and native_test is not None:
                n_val = int(total * val_ratio)
                order = np.random.default_rng(seed).permutation(total)
                indices = order[:n_val] if split == "val" else order[n_val:]
            else:
                indices = np.arange(total)
            if shuffle:
                indices = np.random.default_rng(seed).permutation(indices)

            def _build_batch(batch_index: int) -> FlaxBatch:
                start = batch_index * batch_size
                batch_idx = indices[start : start + batch_size]
                xs: list[npt.ArrayLike] = []
                ys: list[npt.ArrayLike] = []
                for i in batch_idx:
                    sample = source_dataset[int(i)]
                    if not isinstance(sample, tuple) or len(sample) < 2:
                        raise TypeError(
                            "Expected native dataset samples to be (feature, label) pairs"
                        )
                    sample_t = cast("tuple[object, ...]", sample)
                    img, lbl = sample_t[0], sample_t[1]
                    if isinstance(img, SupportsNumpy):
                        img = img.numpy()
                    xs.append(cast("npt.ArrayLike", img))
                    # A generic target may be a class index (0-d), a segmentation
                    # mask (2-D), or a float regression value — preserved by stack.
                    ys.append(cast("npt.ArrayLike", lbl))
                batch_x = cast("npt.NDArray[np.object_]", np.stack([np.asarray(s) for s in xs]))
                batch_y = np.stack([np.asarray(v) for v in ys])
                if np.issubdtype(batch_y.dtype, np.integer):
                    batch_y = batch_y.astype(np.int64)
                if transform_fn is not None:
                    batch_x = cast(
                        "npt.NDArray[np.object_]",
                        np.stack([transform_fn(cast("npt.ArrayLike", s)) for s in batch_x]),
                    )
                built = FlaxBatch(features=as_jax_array(batch_x), labels=as_jax_array(batch_y))
                if batch_transform_fn is not None:
                    f, l = batch_transform_fn(built.features, built.labels)
                    built = FlaxBatch(features=f, labels=l)
                return built

            n_batches = -(-len(indices) // batch_size)  # ceil division
            return _LazyFlaxBatchStream(_build_batch, n_batches)

        if shuffle:
            rng_np = np.random.default_rng(seed)
            order = rng_np.permutation(len(x))
            x, y = x[order], y[order]

        batches: list[FlaxBatch] = []
        for start in range(0, len(x), batch_size):
            batch_x = x[start : start + batch_size]
            batch_y = y[start : start + batch_size]
            if transform_fn is not None:
                batch_x = cast(
                    "npt.NDArray[np.object_]",
                    np.stack([transform_fn(cast("npt.ArrayLike", s)) for s in batch_x]),
                )
            feat = as_jax_array(batch_x)
            lbl = as_jax_array(batch_y)
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
        transform_fn: ArrayTransform | None = None,
        batch_transform_fn: BatchTransform | None = None,
        sequence_length: int | None = None,
    ) -> list[FlaxBatch]:
        """Route to a FLAX-native dataset when ``_native_train_flax`` is set.

        The native FLAX path stores ``list[FlaxBatch]`` at a native batch size
        chosen by the loader. This helper flattens that list to samples, then
        applies the caller-requested split/shuffle/batch semantics plus
        optional transforms so val/test stay deterministic.
        """
        import jax.numpy as jnp

        from dagnam.data.loaders.flax import FlaxBatch

        as_jax_array = cast("JaxArrayFactory", jnp.asarray)

        native_train_flax = self.native_train_flax
        native_test_flax = self.native_test_flax

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
        features_list: list[npt.NDArray[Any]] = [np.asarray(b.features) for b in source_batches]
        labels_list: list[npt.NDArray[np.int64]] = [np.asarray(b.labels) for b in source_batches]
        # Text features need fixed-length integer tokens before np.concatenate /
        # jnp.asarray: raw strings can't index an Embedding (G078); and sequence
        # rows must share a length or np.concatenate(axis=0) fails. The ragged
        # case is NOT only object-dtype-within-a-batch — each FlaxBatch is often
        # internally rectangular but DIFFERENT batches have different sequence
        # lengths (e.g. 4816 vs 3819), which my earlier object-only guard missed
        # (G079). Pad/truncate every batch to one length whenever the batches
        # can't be concatenated as-is.
        if features_list and self._is_text_features(features_list[0]):
            target = sequence_length or 200
            features_list = [
                self._tokenize_text(list(batch), maxlen=target) for batch in features_list
            ]
        elif features_list and self._batches_need_padding(features_list):
            # ``len(row)`` works for both an object batch (rows are lists) and a
            # rectangular 2-D batch (iterating yields per-sample rows).
            target = sequence_length or max(
                (len(row) for batch in features_list for row in batch), default=1
            )
            features_list = [
                self._pad_sequences(cast("Sequence[Sequence[int]]", list(batch)), maxlen=target)
                for batch in features_list
            ]
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
                chunk_x = cast(
                    "npt.NDArray[np.object_]",
                    np.stack([transform_fn(cast("npt.ArrayLike", s)) for s in chunk_x]),
                )
            feat = as_jax_array(chunk_x)
            lbl = as_jax_array(chunk_y)
            batch = FlaxBatch(features=feat, labels=lbl)
            if batch_transform_fn is not None:
                f, l = batch_transform_fn(batch.features, batch.labels)
                batch = FlaxBatch(features=f, labels=l)
            batches.append(batch)
        return batches

    def native_flax_dataset(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float = 0.1,
        seed: int = 42,
        transform_fn: ArrayTransform | None = None,
        batch_transform_fn: BatchTransform | None = None,
    ) -> list[FlaxBatch]:
        """Build the split as a list of native Flax/JAX batches (public wrapper)."""
        return self._native_flax_dataset(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            seed=seed,
            transform_fn=transform_fn,
            batch_transform_fn=batch_transform_fn,
        )

    def _try_upgrade_to_native_flax(self) -> bool:
        """Compatibility hook for already-populated native Flax splits."""
        return getattr(self, "_native_train_flax", None) is not None

    def to_flax_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        image_size: tuple[int, int] = (224, 224),
        column_roles: dict[str, str] | None = None,
        binding: dict[str, Any] | None = None,
        transform_fn: ArrayTransform | None = None,
        batch_transform_fn: BatchTransform | None = None,
        vocab_size: int | None = None,
        sequence_length: int | None = None,
    ) -> list[FlaxBatch] | _LazyFlaxBatchStream:
        """Create a sequence of Flax batches for the specified split.

        Supports tabular (CSV/TSV/JSON/JSONL), image-folder, and audio-folder
        datasets, plus system datasets (via the native path).

        Args:
            column_roles: Optional mapping of column names to roles for
                tabular datasets. Ignored for image/audio formats.
            image_size: Target ``(height, width)`` for user image-folder
                datasets. Ignored by other dataset formats.

        Raises ImportError if jax/flax is not installed.
        Raises ValueError for unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
        if column_roles is None and binding is not None:
            column_roles = _column_roles_from_binding(binding)

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
            import jax

            del jax
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
                sequence_length=sequence_length,
            )
        if self._native_train is not None:
            return self._native_to_flax(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                transform_fn=transform_fn,
                batch_transform_fn=batch_transform_fn,
                vocab_size=vocab_size,
                sequence_length=sequence_length,
            )

        fmt = self.format.lower()

        # Image folder
        if fmt == "image_folder":
            from dagnam.data.loaders.image_folder import (
                ImageTransform,
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
                image_size=image_size,
                transform_fn=cast("ImageTransform | None", transform_fn),
                batch_transform_fn=batch_transform_fn,
            )

        # Audio folder
        if fmt == "audio_folder" or (
            fmt not in ("csv", "tsv", "json", "jsonl") and self.dataset_type == "audio"
        ):
            from dagnam.data.loaders.audio.transforms import (
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
            )  # pragma: no cover -- defensive: unsupported formats already rejected by this method's earlier guard (L309-310)

        from dagnam.data.loaders.flax import (
            BatchTransform as TabularBatchTransform,
            FeatureTransform,
            create_flax_dataset,
        )

        return create_flax_dataset(
            dagnam_ds=self,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            column_roles=column_roles,
            binding=binding,
            transform_fn=cast("FeatureTransform | None", transform_fn),
            batch_transform_fn=cast("TabularBatchTransform | None", batch_transform_fn),
        )
