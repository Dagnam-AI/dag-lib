"""Image folder dataset loader for PyTorch.

Loads image classification datasets organized in class-folder structure:
- Split layout: root/{split}/{class}/*.jpg
- Unsplit layout: root/{class}/*.jpg (uses deterministic splits)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dagnam.data.loaders.media_utils import (
    discover_class_folders,
    ensure_extracted,
    split_indices,
)

if TYPE_CHECKING:
    from dagnam.data.dataset import DagnamDataset
    from torch.utils.data import DataLoader


def create_pytorch_loader(
    dagnam_ds: "DagnamDataset",
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
) -> "DataLoader":
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

    # Build transforms
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if layout.has_explicit_splits:
        # Use explicit split directories
        # Normalize split name: 'val' -> check for 'val' or 'validation'
        split_dir = _resolve_split_dir(data_root, split, layout.splits)
        dataset = datasets.ImageFolder(str(split_dir), transform=transform)
    else:
        # Unsplit: load all images and use deterministic subset
        dataset = datasets.ImageFolder(str(data_root), transform=transform)
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
        pin_memory=True,
        drop_last=(split == "train"),
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
        f"No directory found for split '{split}' in {root}. "
        f"Available splits: {available_splits}"
    )
