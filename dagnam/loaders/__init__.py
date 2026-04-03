"""Loaders package — format-specific dataset factories.

Available loaders (imported lazily to avoid requiring framework deps at
import time):

- ``dagnam.loaders.csv_loader.create_pytorch_loader``  — CSV/TSV → PyTorch
- ``dagnam.loaders.json_loader.create_pytorch_loader`` — JSON/JSONL → PyTorch
- ``dagnam.loaders.tf_loader.create_tensorflow_dataset`` — tabular → tf.data
- ``dagnam.loaders.flax_loader.create_flax_dataset``   — tabular → JAX batches

These are dispatched automatically by the corresponding
``DagnamDataset.to_*`` methods based on the dataset format.
"""
