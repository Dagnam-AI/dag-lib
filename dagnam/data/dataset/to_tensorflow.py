"""TensorFlow conversion for DagnamDataset."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, SupportsInt, cast

import numpy as np
import numpy.typing as npt

from dagnam._types import IndexedDataset, SupportsNumpy, TensorflowDataset, TensorflowModule
from dagnam.data._polars_utils import clamp_token_ids, column_roles_from_binding
from dagnam.data.dataset._typing import DatasetMixinBase

TensorflowMapFn = Callable[..., object]


def _cardinality_to_int(value: object) -> int:
    if isinstance(value, SupportsInt):
        return int(value)
    raise TypeError("TensorFlow cardinality did not return an integer-compatible value")


def _iter_native_samples(
    read_fn: Callable[[int], tuple[npt.NDArray[Any], npt.NDArray[Any]]],
    indices: npt.NDArray[Any],
) -> Iterator[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
    """Lazily yield ``(feature, label)`` one sample at a time per split index.

    Backs the ``tf.data.Dataset.from_generator`` stream for torchvision-style
    native datasets so the whole split is never materialized at once. Kept at
    module scope (rather than a closure) so it is directly unit-testable —
    ``from_generator`` runs it inside tf.data's own thread, which the coverage
    tracer cannot follow.
    """
    for idx in indices:
        yield read_fn(int(idx))


class TensorflowDatasetMixin(DatasetMixinBase):
    """Tensorflow conversion methods."""

    def _native_to_tensorflow(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float,
        seed: int,
        map_fn: TensorflowMapFn | None = None,
        batch_map_fn: TensorflowMapFn | None = None,
        vocab_size: int | None = None,
        sequence_length: int | None = None,
    ) -> TensorflowDataset:
        """Convert a torchvision-style native dataset into a tf.data.Dataset.

        Materializes all samples in memory as numpy arrays then constructs a
        ``tf.data.Dataset.from_tensor_slices``. Intended for small benchmark
        datasets (MNIST, CIFAR-10, Fashion-MNIST). For larger datasets the
        caller should use ``_native_train_tf`` / ``_native_test_tf`` set by
        the TF-specific system loader (see ``_load_native_tf``).
        """
        import tensorflow as tf

        tf = cast("TensorflowModule", tf)
        # Some tf.data constructors used below (TensorSpec, as_dtype, from_generator)
        # are not on the typed Protocol surface; reach them via an Any view.
        tf_any = cast("Any", tf)
        native_train = self._native_train
        native_test = self._native_test
        if native_train is None:
            raise ValueError("No native dataset is available")

        if isinstance(native_train, tuple):
            # numpy tuple datasets (IMDB)
            x_train, y_train = native_train
            if native_test is not None and isinstance(native_test, tuple):
                x_test, y_test = native_test
            else:
                x_test, y_test = (), ()
            num_words = vocab_size if vocab_size is not None else 20000
            maxlen = sequence_length if sequence_length is not None else 200
            if self._is_text_features(np.asarray(x_train)):
                # Raw text -> fixed-length integer tokens: a keras Embedding
                # cannot cast strings ("Cast string to int32"). Tokenize so the TF
                # model receives integer ids, matching flax/pytorch.
                x_train = self._tokenize_text(list(x_train), maxlen=maxlen, num_words=num_words)
                x_test = self._tokenize_text(list(x_test), maxlen=maxlen, num_words=num_words)
            elif np.asarray(x_train).dtype == object:
                # Ragged (variable-length) integer sequences arrive as object arrays;
                # pad them. Rectangular numeric arrays keep their natural dtype.
                x_train = self._pad_sequences(
                    cast("Sequence[Sequence[int]]", x_train), maxlen=maxlen, num_words=num_words
                )
                x_test = self._pad_sequences(
                    cast("Sequence[Sequence[int]]", x_test), maxlen=maxlen, num_words=num_words
                )
            elif vocab_size is not None:
                # A rectangular, already-padded token array still carries raw ids
                # that can exceed the embedding's vocab; clamp out-of-vocab ids to 0
                # exactly like the ragged path above, else the embedding raises
                # "index out of range" (G306, mirrors the pytorch fix for G247).
                x_train = clamp_token_ids(np.asarray(x_train), vocab_size)
                x_test = clamp_token_ids(np.asarray(x_test), vocab_size)
            if split == "test":
                x = cast("npt.NDArray[np.object_]", np.asarray(x_test))
                y = np.asarray(y_test).astype(np.int64)
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
            source_ds = tf.data.Dataset.from_tensor_slices((x, y))
            n_items = len(x)
            skip_tf_shuffle = False
        else:
            # torchvision-style: stream samples lazily into tf.data. Do NOT
            # materialize the whole split — large corpora (e.g. Speech Commands
            # audio: ~100k decoded waveforms, several GB) would OOM the smoke
            # check and training. Resolve the split indices from the dataset
            # length, then index the source per-sample via ``from_generator`` so
            # only one batch is resident at a time. The binding/transform
            # ``map_fn`` still runs via ``ds.map`` below, exactly as before, and
            # the (lazy) pytorch path already worked this way.
            source = native_test if (split == "test" and native_test is not None) else native_train
            source_dataset = cast("IndexedDataset", source)
            total = len(source_dataset)
            if split in ("train", "val") and native_test is not None:
                n_val = int(total * val_ratio)
                rng = np.random.default_rng(seed)
                order = rng.permutation(total)
                indices = order[:n_val] if split == "val" else order[n_val:]
            else:
                indices = np.arange(total)
            # Shuffle the cheap index array up front, NOT the decoded samples: a
            # full-size tf.data shuffle buffer over the from_generator stream would
            # force EVERY waveform to be decoded into the buffer before the first
            # batch is yielded (the Speech Commands OOM — ~85k waveforms, several
            # GB). Pre-shuffling indices gives an equivalent random order in O(n)
            # ints, so the tail skips the tf.data shuffle for this lazy branch.
            if shuffle:
                indices = np.random.default_rng(seed).permutation(indices)
            skip_tf_shuffle = True

            def _read_sample(idx: int) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
                sample = source_dataset[idx]
                if not isinstance(sample, tuple) or len(sample) < 2:
                    raise TypeError("Expected native dataset samples to be (feature, label) pairs")
                sample_t = cast("tuple[object, ...]", sample)
                img, lbl = sample_t[0], sample_t[1]
                if isinstance(img, SupportsNumpy):
                    img = img.numpy()
                # A generic target may be a class index (0-d), a segmentation mask
                # (2-D), or a float regression value — np.asarray preserves each;
                # only integer class/mask ids are normalized to int64.
                y_arr = np.asarray(lbl)
                if np.issubdtype(y_arr.dtype, np.integer):
                    y_arr = y_arr.astype(np.int64)
                return np.asarray(img), y_arr

            if len(indices) == 0:
                # Empty split (e.g. val_ratio rounds to 0): an empty, correctly
                # typed dataset keeps downstream batching/iteration working.
                source_ds = tf.data.Dataset.from_tensor_slices(
                    (np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64))
                )
                n_items = 0
            else:
                probe_x, probe_y = _read_sample(int(indices[0]))
                output_signature = (
                    tf_any.TensorSpec(shape=probe_x.shape, dtype=tf_any.as_dtype(probe_x.dtype)),
                    tf_any.TensorSpec(shape=probe_y.shape, dtype=tf_any.as_dtype(probe_y.dtype)),
                )

                source_ds = cast(
                    "TensorflowDataset",
                    tf_any.data.Dataset.from_generator(
                        lambda: _iter_native_samples(_read_sample, indices),
                        output_signature=output_signature,
                    ),
                )
                n_items = len(indices)

        ds = source_ds
        if shuffle and not skip_tf_shuffle:
            ds = ds.shuffle(buffer_size=max(n_items, 1024), seed=seed)
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
        map_fn: TensorflowMapFn | None = None,
        batch_map_fn: TensorflowMapFn | None = None,
        vocab_size: int | None = None,
        sequence_length: int | None = None,
    ) -> TensorflowDataset:
        """Route to a TF-native dataset when ``_native_train_tf`` is set.

        Partitions the native train split into train/val subsets so that
        callers requesting ``split='val'`` get a distinct slice instead of the
        full training set.
        """
        import tensorflow as tf

        tf = cast("TensorflowModule", tf)
        native_train_tf = self.native_train_tf
        native_test_tf = self.native_test_tf

        if split == "test":
            ds = native_test_tf if native_test_tf is not None else native_train_tf
        elif split == "val":
            if native_train_tf is None:
                raise ValueError("No native TF dataset available for 'val' split")
            cardinality = _cardinality_to_int(
                tf.data.experimental.cardinality(native_train_tf).numpy()
            )
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
            cardinality = _cardinality_to_int(
                tf.data.experimental.cardinality(native_train_tf).numpy()
            )
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY or cardinality < 0:
                cardinality = sum(1 for _ in native_train_tf)
            n_val = max(1, int(cardinality * val_ratio))
            ds = native_train_tf.skip(n_val)

        if ds is None:
            raise ValueError(f"No native TF dataset for split '{split}'")

        # Tokenize raw-text features: tfds text datasets (imdb_reviews,
        # wikitext) yield a tf.data of (string, label); an integer Embedding can't
        # cast strings. Map each string to fixed-length int tokens (same crc32
        # scheme as flax/pytorch via _tokenize_text) when the feature dtype is
        # string and a sequence_length was provided.
        tf_any = cast("Any", tf)
        element_spec = cast("Any", ds).element_spec
        feature_spec = element_spec[0] if isinstance(element_spec, tuple) else None
        if (
            sequence_length is not None
            and feature_spec is not None
            and getattr(feature_spec, "dtype", None) == tf_any.string
        ):
            maxlen = sequence_length
            num_words = vocab_size if vocab_size is not None else 20000
            # Materialize the (string, label) split, tokenize with the shared crc32
            # scheme (identical tokens to flax/pytorch), and rebuild a numeric
            # tf.data. Done eagerly in numpy rather than a tf.py_function map so the
            # tokens match the other frameworks exactly and the path is verifiable.
            rows = list(cast("Any", ds).as_numpy_iterator())
            texts = [
                feat.decode("utf-8") if isinstance(feat, bytes) else str(feat) for feat, _ in rows
            ]
            token_labels = np.asarray([label for _, label in rows])
            tokens = self._tokenize_text(texts, maxlen=maxlen, num_words=num_words)
            ds = tf.data.Dataset.from_tensor_slices((tokens, token_labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=max(batch_size * 16, 1024), seed=seed)
        if map_fn is not None:
            ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if batch_map_fn is not None:
            ds = ds.map(batch_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def native_tensorflow_dataset(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        val_ratio: float = 0.1,
        seed: int = 42,
        map_fn: TensorflowMapFn | None = None,
        batch_map_fn: TensorflowMapFn | None = None,
    ) -> TensorflowDataset:
        """Build the split as a native ``tf.data.Dataset`` (public wrapper)."""
        return self._native_tensorflow_dataset(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            val_ratio=val_ratio,
            seed=seed,
            map_fn=map_fn,
            batch_map_fn=batch_map_fn,
        )

    def _try_upgrade_to_native_tf(self) -> bool:
        """Compatibility hook for already-populated native TensorFlow splits."""
        return getattr(self, "_native_train_tf", None) is not None

    def to_tensorflow_dataset(
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
        map_fn: TensorflowMapFn | None = None,
        batch_map_fn: TensorflowMapFn | None = None,
        vocab_size: int | None = None,
        sequence_length: int | None = None,
    ) -> TensorflowDataset:
        """Create a TensorFlow Dataset for the specified split.

        Supports tabular (CSV/TSV/JSON/JSONL), image-folder, and audio-folder
        datasets, plus system datasets (via the native path).

        Args:
            column_roles: Optional mapping of column names to roles for
                tabular datasets (``{"x": "feature", "label": "target"}``).
                Ignored for image/audio formats.
            image_size: Target ``(height, width)`` for user image-folder
                datasets. Ignored by other dataset formats.

        Raises ImportError if tensorflow is not installed.
        Raises ValueError for unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
        if column_roles is None and binding is not None:
            column_roles = column_roles_from_binding(binding)

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
            import tensorflow

            del tensorflow
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
                vocab_size=vocab_size,
                sequence_length=sequence_length,
            )
        if self._native_train is not None:
            return self._native_to_tensorflow(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                val_ratio=val_ratio,
                seed=seed,
                map_fn=map_fn,
                batch_map_fn=batch_map_fn,
                vocab_size=vocab_size,
                sequence_length=sequence_length,
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
                image_size=image_size,
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
            raise ValueError(
                f"Unsupported format for TensorFlow dataset: {self.format}"
            )  # pragma: no cover -- defensive: unsupported formats already rejected by this method's earlier guard (L285-286)

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
            binding=binding,
            map_fn=map_fn,
            batch_map_fn=batch_map_fn,
        )
