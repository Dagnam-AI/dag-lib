"""PyTorch dataset and collation hook helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence, Sized
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset as _TorchDataset
    from torch.utils.data.sampler import Sampler
else:

    class _TorchDataset:
        @classmethod
        def __class_getitem__(cls, _item: object) -> type[_TorchDataset]:
            return cls


CollateFn = Callable[[object], object]


class LoaderLike(Protocol):
    @property
    def dataset(self) -> object: ...

    @property
    def batch_size(self) -> int | None: ...

    @property
    def sampler(self) -> object: ...

    @property
    def num_workers(self) -> int: ...

    @property
    def pin_memory(self) -> bool: ...

    @property
    def drop_last(self) -> bool: ...

    @property
    def timeout(self) -> float: ...

    @property
    def worker_init_fn(self) -> object: ...

    @property
    def multiprocessing_context(self) -> object: ...

    @property
    def generator(self) -> object: ...

    @property
    def prefetch_factor(self) -> int | None: ...

    @property
    def persistent_workers(self) -> bool: ...

    @property
    def pin_memory_device(self) -> str: ...


def _wrap_collate(
    collate_fn: CollateFn | None = None,
    batch_transform: CollateFn | None = None,
) -> CollateFn | None:
    """Apply an optional batch transform after PyTorch collation."""
    if batch_transform is None:
        return collate_fn

    def wrapped(batch: object) -> object:
        if collate_fn is None:
            from torch.utils.data._utils import collate as collate_module

            default_collate_fn = cast("CollateFn", collate_module.default_collate)
            collated = default_collate_fn(batch)
        else:
            collated = collate_fn(batch)
        return batch_transform(collated)

    return wrapped


def _with_collate(
    loader: LoaderLike,
    collate_fn: CollateFn | None = None,
    batch_transform: CollateFn | None = None,
) -> DataLoader[object]:
    """Rebuild a DataLoader with hook-aware collation when needed."""
    wrapped_collate = _wrap_collate(collate_fn, batch_transform)
    if wrapped_collate is None:
        return cast("DataLoader[object]", loader)

    from torch.utils.data import DataLoader

    return DataLoader(
        cast("_TorchDataset[object]", loader.dataset),
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=cast("Sampler[object] | Iterable[object] | None", loader.sampler),
        num_workers=loader.num_workers,
        collate_fn=wrapped_collate,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=cast("Callable[[int], None] | None", loader.worker_init_fn),
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )


class _TransformDataset(_TorchDataset[object]):
    """Map-style dataset wrapper that applies sample and target hooks."""

    def __init__(
        self,
        dataset: _TorchDataset[object] | Sequence[object],
        transform: Callable[[object], object] | None = None,
        target_transform: Callable[[object], object] | None = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(cast("Sized", self.dataset))

    @override
    def __getitem__(self, index: int) -> object:
        item = self.dataset[index]
        if isinstance(item, tuple):
            item_tuple = cast("tuple[object, ...]", item)
            if len(item_tuple) < 2:
                return self.transform(item_tuple) if self.transform is not None else item_tuple
            data = item_tuple[0]
            target = item_tuple[1]
            rest = item_tuple[2:]
            if self.transform is not None:
                data = self.transform(data)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if rest:
                return (data, target, *rest)
            return data, target
        if self.transform is not None:
            return self.transform(item)
        return item


class _ChannelsFirstImageDataset(_TorchDataset[object]):
    """Transpose an image sample from channels-last to channels-first for PyTorch.

    The decoder + transform pipeline produces canonical channels-last ``[H, W, C]``
    images (numpy/PIL convention, which TF/Flax consume as-is); only PyTorch needs
    ``[C, H, W]``, applied here per item before the default collate stacks the batch.
    """

    def __init__(self, dataset: _TorchDataset[object] | Sequence[object]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(cast("Sized", self.dataset))

    @override
    def __getitem__(self, index: int) -> object:
        item = self.dataset[index]
        if isinstance(item, tuple):
            item_tuple = cast("tuple[object, ...]", item)
            if len(item_tuple) >= 2:
                data = np.moveaxis(np.asarray(item_tuple[0]), -1, 0)
                return (data, *item_tuple[1:])
        return item


__all__ = [
    "_ChannelsFirstImageDataset",
    "_TransformDataset",
    "_with_collate",
    "_wrap_collate",
]
