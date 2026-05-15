"""Image folder dataset loader for PyTorch.

Loads image classification datasets organized in class-folder structure:
- Split layout: root/{split}/{class}/*.jpg
- Unsplit layout: root/{class}/*.jpg (uses deterministic splits)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dagnam.data.loaders.media import (
    discover_class_folders,
    ensure_extracted,
    split_indices,
)
from dagnam.data.loaders.torch_utils import should_pin_memory

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from dagnam.data.dataset import DagnamDataset


def create_pytorch_loader(
    dagnam_ds: DagnamDataset,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
    transform=None,
    target_transform=None,
    collate_fn=None,
) -> DataLoader:
    """Create a PyTorch DataLoader from an image-folder dataset.

    Requires ``torchvision`` to be installed.

    Args:
        dagnam_ds: The DagnamDataset instance.
        split: One of 'train', 'val', 'test'.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle. Defaults to True for train.
        val_ratio: Fraction for validation when using deterministic splits.
        test_ratio: Fraction for test when using deterministic splits.
        seed: Random seed for deterministic splitting.
        image_size: Target (height, width) for resizing images.

    Returns:
        A PyTorch DataLoader yielding (image_tensor, label) batches.

    Raises:
        ImportError: If torch or torchvision is not installed.
        FileNotFoundError: If the split directory doesn't exist.
    """
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        raise ImportError(
            "PyTorch is required for image folder loading. "
            "Install with: pip install dagnam[pytorch]"
        )

    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError(
            "torchvision is required for image folder loading. "
            "Install with: pip install torchvision"
        )

    if shuffle is None:
        shuffle = split == "train"

    # Ensure archives are extracted
    data_root = ensure_extracted(dagnam_ds._data_dir)

    # Discover folder layout
    layout = discover_class_folders(data_root)

    # Build default transforms. Augmentation/normalization are explicit hooks.
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    if layout.has_explicit_splits:
        # Use explicit split directories
        # Normalize split name: 'val' -> check for 'val' or 'validation'
        split_dir = _resolve_split_dir(data_root, split, layout.splits)
        dataset = datasets.ImageFolder(
            str(split_dir),
            transform=transform,
            target_transform=target_transform,
        )
    else:
        # Unsplit: load all images and use deterministic subset
        dataset = datasets.ImageFolder(
            str(data_root),
            transform=transform,
            target_transform=target_transform,
        )
        n = len(dataset)
        train_idx, val_idx, test_idx = split_indices(
            n, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        dataset = Subset(dataset, split_map[split])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=should_pin_memory(),
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )


def _resolve_split_dir(root: Path, split: str, available_splits: list[str]) -> Path:
    """Resolve the actual directory for a requested split name.

    Handles aliases like 'val' -> 'validation' and vice versa.
    """
    # Direct match
    if split in available_splits:
        return root / split

    # Alias mapping
    aliases = {
        "val": ["validation", "dev"],
        "validation": ["val"],
        "test": ["dev"],
    }

    for alias in aliases.get(split, []):
        if alias in available_splits:
            return root / alias

    # Fallback: if requesting val/test but only train exists, return train
    # (the caller will get a subset of train data)
    if "train" in available_splits:
        return root / "train"

    raise FileNotFoundError(
        f"No directory found for split '{split}' in {root}. Available splits: {available_splits}"
    )


def create_tensorflow_dataset(
    dagnam_ds: DagnamDataset,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
    map_fn=None,
    batch_map_fn=None,
):
    """Create a tf.data.Dataset from an image-folder dataset.

    Uses ``tf.keras.utils.image_dataset_from_directory`` under the hood. When
    the dataset has explicit split directories (``root/{train,val,test}/...``),
    the corresponding directory is loaded. When the dataset is unsplit, a
    deterministic train/val/test partition is computed per the ``seed`` and
    ``val_ratio``/``test_ratio`` arguments.

    Args:
        map_fn: Per-sample map function ``(image, label) -> (image, label)``.
        batch_map_fn: Post-batch map function applied to whole batches.
    """
    import tensorflow as tf

    if shuffle is None:
        shuffle = split == "train"

    data_root = ensure_extracted(dagnam_ds._data_dir)
    layout = discover_class_folders(data_root)

    if layout.has_explicit_splits:
        split_dir = _resolve_split_dir(data_root, split, layout.splits)
        ds = tf.keras.utils.image_dataset_from_directory(
            str(split_dir),
            labels="inferred",
            label_mode="int",
            batch_size=None,
            image_size=image_size,
            shuffle=False,
            seed=seed,
        )
    else:
        # Use Keras' validation_split API for a train/val cut, then further
        # split the remaining train into a deterministic test subset.
        # For simplicity and robustness we materialize all samples then apply
        # a stable index partition.
        full = tf.keras.utils.image_dataset_from_directory(
            str(data_root),
            labels="inferred",
            label_mode="int",
            batch_size=None,
            image_size=image_size,
            shuffle=False,
            seed=seed,
        )
        # Count cardinality
        n = tf.data.experimental.cardinality(full).numpy()
        if n == tf.data.experimental.UNKNOWN_CARDINALITY or n < 0:
            # Fall back to Python iteration for count
            n = sum(1 for _ in full)
        train_idx, val_idx, test_idx = split_indices(
            int(n), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        keep_indices = set(split_map[split])
        keep_mask = [i in keep_indices for i in range(int(n))]
        keep_tensor = tf.constant(keep_mask, dtype=tf.bool)
        ds = (
            full.enumerate()
            .filter(lambda i, _payload: keep_tensor[i])
            .map(lambda i, payload: payload)
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=max(batch_size * 16, 1024), seed=seed)

    if map_fn is not None:
        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    if batch_map_fn is not None:
        ds = ds.map(batch_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def create_flax_dataset(
    dagnam_ds: DagnamDataset,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
    transform_fn=None,
    batch_transform_fn=None,
) -> list:
    """Create a list of FlaxBatch from an image-folder dataset.

    Reads all images for the split into memory as JAX arrays. For very large
    image datasets prefer ``to_tensorflow_dataset`` (streamed) and convert
    to JAX per batch in the training loop instead.
    """
    import jax.numpy as jnp
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for FLAX image-folder loading. "
            "Install with: pip install dagnam[flax] Pillow"
        )

    from dagnam.data.loaders.flax import FlaxBatch

    if shuffle is None:
        shuffle = split == "train"

    data_root = ensure_extracted(dagnam_ds._data_dir)
    layout = discover_class_folders(data_root)

    if layout.has_explicit_splits:
        split_dir = _resolve_split_dir(data_root, split, layout.splits)
        samples, classes = _gather_image_samples(split_dir)
    else:
        samples, classes = _gather_image_samples(data_root)
        train_idx, val_idx, test_idx = split_indices(
            len(samples), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        samples = [samples[i] for i in split_map[split]]

    if shuffle:
        import random as _random

        rng = _random.Random(seed)
        rng.shuffle(samples)

    batches: list[FlaxBatch] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = []
        labels = []
        for path, label in chunk:
            img = Image.open(path).convert("RGB").resize(image_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            if transform_fn is not None:
                arr = transform_fn(arr)
            images.append(arr)
            labels.append(label)
        x = jnp.asarray(np.stack(images))
        y = jnp.asarray(np.array(labels, dtype=np.int64))
        batch = FlaxBatch(features=x, labels=y)
        if batch_transform_fn is not None:
            feat, lbl = batch_transform_fn(batch.features, batch.labels)
            batch = FlaxBatch(features=feat, labels=lbl)
        batches.append(batch)

    return batches


def _gather_image_samples(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    """Enumerate (image_path, class_idx) pairs sorted by class name, then filename."""
    classes = sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    samples: list[tuple[Path, int]] = []
    for cls in classes:
        cls_dir = root / cls
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                samples.append((p, class_to_idx[cls]))
    return samples, classes
