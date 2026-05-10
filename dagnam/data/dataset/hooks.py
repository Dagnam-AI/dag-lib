"""PyTorch dataset and collation hook helpers."""

from __future__ import annotations


def _wrap_collate(collate_fn=None, batch_transform=None):
    """Apply an optional batch transform after PyTorch collation."""
    if batch_transform is None:
        return collate_fn

    def wrapped(batch):
        if collate_fn is None:
            from torch.utils.data._utils.collate import default_collate

            collated = default_collate(batch)
        else:
            collated = collate_fn(batch)
        return batch_transform(collated)

    return wrapped


def _with_collate(loader, collate_fn=None, batch_transform=None):
    """Rebuild a DataLoader with hook-aware collation when needed."""
    wrapped_collate = _wrap_collate(collate_fn, batch_transform)
    if wrapped_collate is None:
        return loader

    from torch.utils.data import DataLoader

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        collate_fn=wrapped_collate,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )


class _TransformDataset:
    """Map-style dataset wrapper that applies sample and target hooks."""

    def __init__(self, dataset, transform=None, target_transform=None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, tuple) and len(item) >= 2:
            data = item[0]
            target = item[1]
            rest = item[2:]
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


__all__ = ["_TransformDataset", "_with_collate", "_wrap_collate"]
