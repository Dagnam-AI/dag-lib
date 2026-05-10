"""JSON/JSONL loader — converts JSON/JSONL data into PyTorch DataLoaders.

Delegates to the same splitting, encoding, and DataLoader creation logic
as the CSV loader. Format-specific parsing is handled by
``DagnamDataset.to_pandas()``.
"""

from dagnam.data.loaders.csv import create_pytorch_loader

__all__ = ["create_pytorch_loader"]
