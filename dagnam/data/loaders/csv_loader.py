"""Compatibility wrapper for ``dagnam.data.loaders.csv``."""

from dagnam.data.loaders.csv import *
from dagnam.data.loaders.csv import (
    FEATURE_ROLES,
    TARGET_ROLES,
    create_pytorch_loader,
    detect_label_column,
    split_by_roles,
)

__all__ = [
    "FEATURE_ROLES",
    "TARGET_ROLES",
    "create_pytorch_loader",
    "detect_label_column",
    "split_by_roles",
]
