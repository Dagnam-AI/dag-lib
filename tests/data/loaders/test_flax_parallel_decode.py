"""Order/determinism safety net for the parallel per-chunk decode in flax.py.

These tests pin the contract that parallelizing the per-sample decode must
preserve: identical batch order, identical labels, and identical seeded-shuffle
behavior. They pass on the sequential implementation and must keep passing once
``build_flax_batches`` decodes a chunk with a thread pool.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from dagnam.data.loaders.flax import FlaxBatch, build_flax_batches

if TYPE_CHECKING:
    import jax


def _fake_loader(sample: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Materialize a sample to ``(feature_row, label)`` keyed by its value."""
    value, label = sample
    return np.full((4,), float(value), dtype=np.float32), label


class TestParallelDecodeDeterminism:
    def test_preserves_input_order_without_shuffle(self) -> None:
        samples = [(i, i % 3) for i in range(10)]
        batches = build_flax_batches(samples, batch_size=4, load_sample=_fake_loader)

        features = np.concatenate([np.asarray(b.features) for b in batches])
        labels = np.concatenate([np.asarray(b.labels) for b in batches])
        assert [int(row[0]) for row in features] == list(range(10))
        assert labels.tolist() == [i % 3 for i in range(10)]

    def test_labels_track_their_features(self) -> None:
        samples = [(i * 11, i) for i in range(7)]
        batches = build_flax_batches(samples, batch_size=3, load_sample=_fake_loader)
        for batch in batches:
            features = np.asarray(batch.features)
            labels = np.asarray(batch.labels)
            for row, label in zip(features, labels, strict=True):
                assert int(row[0]) == int(label) * 11

    def test_seeded_shuffle_is_stable(self) -> None:
        samples = [(i, i) for i in range(12)]
        a = build_flax_batches(
            samples, batch_size=12, load_sample=_fake_loader, shuffle=True, seed=5
        )
        b = build_flax_batches(
            samples, batch_size=12, load_sample=_fake_loader, shuffle=True, seed=5
        )
        c = build_flax_batches(
            samples, batch_size=12, load_sample=_fake_loader, shuffle=True, seed=6
        )
        assert np.array_equal(np.asarray(a[0].labels), np.asarray(b[0].labels))
        assert not np.array_equal(np.asarray(a[0].labels), np.asarray(c[0].labels))

    def test_batch_transform_still_applied(self) -> None:
        samples = [(1, 0), (2, 1), (3, 0)]

        def _double(features: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
            return jnp.asarray(features) * 2, labels

        batches = build_flax_batches(
            samples, batch_size=2, load_sample=_fake_loader, batch_transform_fn=_double
        )
        first = np.asarray(batches[0].features)
        assert int(first[0][0]) == 2  # value 1 doubled
        assert int(first[1][0]) == 4  # value 2 doubled

    def test_empty_input_returns_no_batches(self) -> None:
        assert build_flax_batches([], batch_size=4, load_sample=_fake_loader) == []

    def test_returns_flax_batches(self) -> None:
        samples = [(i, 0) for i in range(5)]
        batches = build_flax_batches(samples, batch_size=2, load_sample=_fake_loader)
        assert all(isinstance(b, FlaxBatch) for b in batches)

    def test_decode_runs_on_multiple_threads(self) -> None:
        """The per-chunk decode fans out across worker threads."""
        seen: set[int] = set()
        lock = threading.Lock()

        def _record(sample: tuple[int, int]) -> tuple[np.ndarray, int]:
            with lock:
                seen.add(threading.get_ident())
            return np.full((4,), float(sample[0]), dtype=np.float32), sample[1]

        samples = [(i, 0) for i in range(64)]
        build_flax_batches(samples, batch_size=64, load_sample=_record)
        assert len(seen) >= 2
