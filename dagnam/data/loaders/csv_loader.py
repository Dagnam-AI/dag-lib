"""Compatibility wrapper for ``dagnam.data.loaders.csv``."""

from dagnam.data.loaders.csv import *
from dagnam.data.loaders.csv import (
    _FEATURE_ROLES,
    _TARGET_ROLES,
    _detect_label_column,
    _split_by_roles,
)
