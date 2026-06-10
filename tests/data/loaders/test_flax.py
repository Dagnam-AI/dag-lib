"""Coverage for dagnam.data.loaders.flax.build_flax_batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from dagnam.data.loaders.flax import FlaxBatch, build_flax_batches

if TYPE_CHECKING:
    import jax


def _load(sample: tuple[int, int]) -> tuple[np.ndarray, int]:
    value, label = sample
    return np.full((3,), float(value), dtype=np.float32), label


def test_build_flax_batches_batches_and_keeps_partial() -> None:
    samples = [(i, i % 2) for i in range(5)]
    batches = build_flax_batches(samples, batch_size=2, load_sample=_load)
    assert len(batches) == 3  # 2 + 2 + 1
    assert all(isinstance(b, FlaxBatch) for b in batches)
    assert batches[0].features.shape == (2, 3)
    assert batches[-1].features.shape == (1, 3)


def test_build_flax_batches_empty_input() -> None:
    assert build_flax_batches([], batch_size=2, load_sample=_load) == []


def test_build_flax_batches_shuffle_is_seeded() -> None:
    samples = [(i, i) for i in range(6)]
    a = build_flax_batches(samples, batch_size=6, load_sample=_load, shuffle=True, seed=1)
    b = build_flax_batches(samples, batch_size=6, load_sample=_load, shuffle=True, seed=1)
    c = build_flax_batches(samples, batch_size=6, load_sample=_load, shuffle=True, seed=2)
    assert np.array_equal(np.asarray(a[0].labels), np.asarray(b[0].labels))
    assert not np.array_equal(np.asarray(a[0].labels), np.asarray(c[0].labels))


def test_build_flax_batches_applies_batch_transform() -> None:
    samples = [(1, 0), (2, 1)]

    def _double(features: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.asarray(features) * 2, labels

    batches = build_flax_batches(
        samples, batch_size=2, load_sample=_load, batch_transform_fn=_double
    )
    assert float(np.asarray(batches[0].features)[0][0]) == 2.0  # value 1 doubled
