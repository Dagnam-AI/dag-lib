"""Shared pieces of the recipes of adversarial texts for the token estimate's tests.

Every text is built from code points, repetition and a seeded generator, never kept as text.
Labels are ASCII: a character outside ASCII is written ``U+XXXX``.
"""

from __future__ import annotations

from collections.abc import Callable
import re

type Recipes = list[tuple[str, str]]


def label(kind: str) -> str:
    """``kind`` with every character outside ASCII written as ``U+XXXX``."""
    return re.sub(r"[^\x00-\x7f]", lambda match: f"U+{ord(match.group()):04X}", kind)


def adder(out: Recipes) -> Callable[[str, str], None]:
    def add(kind: str, text: str) -> None:
        out.append((label(kind), text))

    return add
