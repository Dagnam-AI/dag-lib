"""Recipes for the adversarial texts of the token estimate's tests (see the ``_attacks_*`` modules).

``attacks()`` returns the 213 ``(label, text)`` pairs in a fixed order: 201 recipes of
encodings, structured data, characters, random letters and whitespace, then 12 of held-out style
text built from the natural paragraphs it is given.
"""

from __future__ import annotations

from collections.abc import Mapping

from tests.audit._attacks_chars import characters_and_whitespace
from tests.audit._attacks_common import Recipes, label
from tests.audit._attacks_data import encodings_and_data
from tests.audit._attacks_letters import (
    false_positives_and_whitespace,
    held_out_style,
    random_letters,
)

__all__ = ["Recipes", "attacks", "label", "recipes_without_text"]


def recipes_without_text() -> Recipes:
    """The 201 recipes that need no natural text."""
    return [
        *encodings_and_data(),
        *characters_and_whitespace(),
        *random_letters(),
        *false_positives_and_whitespace(),
    ]


def attacks(texts: Mapping[str, str]) -> Recipes:
    """All 213 recipes; ``texts`` maps a language code to a paragraph in it."""
    return [*recipes_without_text(), *held_out_style(texts)]
