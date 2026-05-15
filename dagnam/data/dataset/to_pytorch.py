"""PyTorch conversion for DagnamDataset."""

from __future__ import annotations

from dagnam.data.dataset.hooks import _TransformDataset, _with_collate, _wrap_collate
from dagnam.data.loaders.torch_utils import should_pin_memory


class PytorchDatasetMixin:
    """Pytorch conversion methods."""

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
    ) -> torch.utils.data.DataLoader:  # noqa: F821
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
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

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
            from dagnam.data.loaders.image_folder import (
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
            from dagnam.data.loaders.audio import (
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
            raise ValueError(f"Unsupported format for PyTorch loader: {self.format}")

        if fmt in ("csv", "tsv"):
            from dagnam.data.loaders.csv import create_pytorch_loader
        else:
            from dagnam.data.loaders.json_array import create_pytorch_loader

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
    ) -> torch.utils.data.DataLoader:  # noqa: F821
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
                native_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
            )
        else:  # train
            n_val = int(len(native_train) * val_ratio)
            n_train = len(native_train) - n_val
            ds, _ = random_split(
                native_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
            )

        if transform is not None or target_transform is not None:
            ds = _TransformDataset(ds, transform, target_transform)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=should_pin_memory(),
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
    ) -> torch.utils.data.DataLoader:  # noqa: F821
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
                y = (
                    torch.from_numpy(np.asarray(y_train[:-n_val] if n_val > 0 else y_train))
                    .float()
                    .unsqueeze(1)
                )
            ds = TensorDataset(x, y)

        if transform is not None or target_transform is not None:
            ds = _TransformDataset(ds, transform, target_transform)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=should_pin_memory(),
            drop_last=(split == "train"),
            collate_fn=collate_fn,
        )
