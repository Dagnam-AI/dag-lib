"""Compatibility wrapper for ``dagnam.resources.checkpoints``."""

from dagnam.resources.checkpoints import *
from dagnam.resources.checkpoints import pick_best, pick_latest

__all__ = ["pick_best", "pick_latest"]
