"""Compatibility wrapper for ``dagnam.resources.training``."""

from dagnam._core.sse import parse_raw_event as parse_event
from dagnam.resources.training import *

__all__ = ["parse_event"]
