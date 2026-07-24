"""PyTorch conversion for DagnamDataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence, Sized
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from dagnam.data.dataset._typing import DatasetMixinBase
from dagnam.data.dataset.hooks import (
    _ChannelsFirstImageDataset,
    _TransformDataset,
    _with_collate,
    _wrap_collate,
)
from dagnam.data.loaders.torch_utils import should_pin_memory

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset

TransformFn = Callable[[object], object]
CollateFn = Callable[[object], object]
TensorFactory = Callable[..., "Tensor"]


def _column_roles_from_binding(binding: dict[str, Any]) -> dict[str, str] | None:
    roles: dict[str, str] = {}
    input_column = binding.get("input_column")
    target_column = binding.get("target_column")
    if isinstance(input_column, str) and input_column:
        roles[input_column] = "feature"
    if isinstance(target_column, str) and target_column:
        roles[target_column] = "target"
    return roles or None


def _audio_binding_options(binding: dict[str, Any] | None) -> tuple[bool, int | None]:
    """Return whether a binding requests raw audio and its fixed sample count."""
    if not isinstance(binding, dict):
        return False, None
    transform = binding.get("input_transform")
    if not isinstance(transform, dict) or transform.get("kind") != "audio":
        return False, None
    params = transform.get("params")
    target_length = params.get("target_length") if isinstance(params, dict) else None
    if not isinstance(target_length, int) or isinstance(target_length, bool) or target_length <= 0:
        target_length = None
    return True, target_length


def _native_target_tensor(
    y: object,
    tensor: TensorFactory,
    torch_long: object,
    torch_float32: object,
) -> Tensor:
    """Shape a native-numpy label array into the correct torch target tensor.

    Integer labels are class indices, so they become a ``[B]`` long tensor — the
    shape/dtype ``nn.CrossEntropyLoss`` expects. Float labels are regression /
    ``BCEWithLogitsLoss`` targets, so they keep the ``[B, 1]`` float form.

    A blanket ``float32 + unsqueeze(1)`` (the old behaviour) produced a ``[B, 1]``
    float target that ``CrossEntropyLoss`` rejects with "0D or 1D target tensor
    expected, multi-target not supported".
    """
    y_arr = np.asarray(y)
    if np.issubdtype(y_arr.dtype, np.integer):
        return tensor(y_arr, dtype=torch_long)
    return tensor(y_arr, dtype=torch_float32).unsqueeze(1)


def _clamp_token_ids(array: npt.NDArray[Any], vocab_size: int) -> npt.NDArray[Any]:
    """Clamp out-of-vocab token ids to 0 so ``nn.Embedding`` never indexes past its table.

    Mirrors ``_pad_sequences``' ``w if w < num_words else 0`` for rectangular
    (already-padded) token arrays, which otherwise pass raw ids straight through and
    crash a vocab-sized embedding with "index out of range" (Round-2 G247).
    """
    return np.where(array < vocab_size, array, 0)


class PytorchDatasetMixin(DatasetMixinBase):
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
        image_size: tuple[int, int] = (224, 224),
        column_roles: dict[str, str] | None = None,
        binding: dict[str, Any] | None = None,
        transform: TransformFn | None = None,
        target_transform: TransformFn | None = None,
        collate_fn: CollateFn | None = None,
        batch_transform: CollateFn | None = None,
        waveform_transform: TransformFn | None = None,
        spectrogram_transform: TransformFn | None = None,
        vocab_size: int | None = None,
    ) -> DataLoader[object]:
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
            image_size: Target ``(height, width)`` for user image-folder
                datasets. Ignored by other dataset formats.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: For unsupported formats or invalid split names.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        try:
            import torch

            del torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for to_pytorch_loader(). "
                "Install with: uv pip install dagnam[pytorch]"
            )

        if shuffle is None:
            shuffle = split == "train"
        if column_roles is None and binding is not None:
            column_roles = _column_roles_from_binding(binding)

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
                vocab_size=vocab_size,
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
                image_size=image_size,
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

            return_waveform, target_length = _audio_binding_options(binding)

            return create_audio_loader(
                dagnam_ds=self,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                target_length=target_length,
                return_waveform=return_waveform,
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
            binding=binding,
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
        transform: TransformFn | None = None,
        target_transform: TransformFn | None = None,
        collate_fn: CollateFn | None = None,
        vocab_size: int | None = None,
    ) -> DataLoader[object]:
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
                vocab_size=vocab_size,
            )

        # Handle torchvision map-style datasets
        if native_train is None:
            raise ValueError("No native PyTorch dataset is available")

        if split == "test":
            ds = cast("Dataset[object]", native_test if native_test is not None else native_train)
        elif split == "val":
            train_dataset = cast("Sized", native_train)
            n_val = int(len(train_dataset) * val_ratio)
            n_train = len(train_dataset) - n_val
            torch_dataset = cast("Dataset[object]", native_train)
            # torch.Generator is public but absent from torch's stub __all__.
            generator = torch.Generator().manual_seed(seed)  # pyright: ignore[reportPrivateImportUsage]
            _, ds = random_split(
                torch_dataset,
                [n_train, n_val],
                generator=generator,
            )
        else:  # train
            train_dataset = cast("Sized", native_train)
            n_val = int(len(train_dataset) * val_ratio)
            n_train = len(train_dataset) - n_val
            torch_dataset = cast("Dataset[object]", native_train)
            # torch.Generator is public but absent from torch's stub __all__.
            generator = torch.Generator().manual_seed(seed)  # pyright: ignore[reportPrivateImportUsage]
            ds, _ = random_split(
                torch_dataset,
                [n_train, n_val],
                generator=generator,
            )

        if transform is not None or target_transform is not None:
            ds = _TransformDataset(ds, transform, target_transform)

        # Canonical decoded images are channels-last [H, W, C]; PyTorch convs need
        # channels-first [C, H, W]. Transpose only when the bound input is an image.
        if getattr(native_train, "input_kind", None) == "image":
            ds = _ChannelsFirstImageDataset(ds)

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
        transform: TransformFn | None = None,
        target_transform: TransformFn | None = None,
        collate_fn: CollateFn | None = None,
        vocab_size: int | None = None,
    ) -> DataLoader[object]:
        """Build a DataLoader from numpy array tuples (e.g. IMDB)."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # torch.{tensor,long,float32} are public APIs but torch's type stub
        # omits them from __all__, so pyright flags reportPrivateImportUsage.
        tensor = cast("TensorFactory", torch.tensor)  # pyright: ignore[reportPrivateImportUsage]
        torch_long = torch.long  # pyright: ignore[reportPrivateImportUsage]
        torch_float32 = torch.float32  # pyright: ignore[reportPrivateImportUsage]

        if not isinstance(self._native_train, tuple):
            raise ValueError("Native numpy loader requires train arrays")
        x_train, y_train = self._native_train
        if isinstance(self._native_test, tuple):
            x_test, y_test = self._native_test
        else:
            x_test, y_test = (), ()
        num_words = vocab_size if vocab_size is not None else 20000

        if split == "test":
            # IMDB sequences are variable-length object arrays — pad them
            if np.asarray(x_test).dtype == object:
                # Ragged (variable-length) sequences arrive as object arrays; pad them.
                # _pad_sequences also clamps token ids into ``num_words``.
                x_test = self._pad_sequences(
                    cast("Sequence[Sequence[int]]", x_test), num_words=num_words
                )
            elif vocab_size is not None:
                # A rectangular, already-padded token array still carries raw ids
                # that can exceed the embedding's vocab; clamp out-of-vocab ids to 0
                # exactly like the ragged path, else nn.Embedding raises "index out
                # of range" (Round-2 G247).
                x_test = _clamp_token_ids(np.asarray(x_test), vocab_size)
            x_t = tensor(np.asarray(x_test), dtype=torch_long)
            y_t = _native_target_tensor(y_test, tensor, torch_long, torch_float32)
            ds = TensorDataset(x_t, y_t)
        else:
            if np.asarray(x_train).dtype == object:
                # Ragged (variable-length) sequences arrive as object arrays; pad them.
                # _pad_sequences also clamps token ids into ``num_words``.
                x_train = self._pad_sequences(
                    cast("Sequence[Sequence[int]]", x_train), num_words=num_words
                )
            elif vocab_size is not None:
                # Rectangular already-padded token ids also need clamping to the
                # embedding's vocab (Round-2 G247); see the test-split note above.
                x_train = _clamp_token_ids(np.asarray(x_train), vocab_size)
            n_val = int(len(x_train) * val_ratio)
            if split == "val":
                # n_val may round to 0 for a small train set; an empty val split
                # must stay empty. x_train[-0:] would return the ENTIRE array,
                # overlapping train and val (TF/Flax already guard this). Use a
                # dtype/shape-preserving empty slice.
                x_val = x_train[-n_val:] if n_val > 0 else x_train[:0]
                y_val = y_train[-n_val:] if n_val > 0 else y_train[:0]
                x = tensor(np.asarray(x_val), dtype=torch_long)
                y = _native_target_tensor(y_val, tensor, torch_long, torch_float32)
            else:
                x = tensor(
                    np.asarray(x_train[:-n_val] if n_val > 0 else x_train),
                    dtype=torch_long,
                )
                y = _native_target_tensor(
                    y_train[:-n_val] if n_val > 0 else y_train,
                    tensor,
                    torch_long,
                    torch_float32,
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
