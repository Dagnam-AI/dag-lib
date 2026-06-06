"""Shared fakes/helpers for the ``dagnam.data.loaders.system.*`` tests.

These are imported by the per-backend ``test_system_*`` modules so the fake
TFDS loaders and small transform/resolve stand-ins are defined once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tests.typing_helpers import JsonObject


def identity_transform(value: object) -> object:
    return value


def fallback_resolve(_meta: JsonObject) -> str:
    return "FB"


def as_numpy(value: object) -> object:
    return value


class FakeTfdsLoader:
    # Label is ``object`` (not ``int``) so tests can feed non-integer labels
    # to exercise the loader's integer-label rejection path.
    def __init__(self, samples: Sequence[tuple[object, object]]) -> None:
        self._samples = list(samples)

    def __call__(self, *_args: object, **_kwargs: object) -> list[tuple[object, object]]:
        return self._samples


class SplitTfdsLoader:
    def __init__(
        self,
        train_samples: list[tuple[object, int]],
        test_samples: list[tuple[object, int]],
    ) -> None:
        self._train_samples = train_samples
        self._test_samples = test_samples

    def __call__(
        self,
        _name: str,
        split: str | None = None,
        _as_supervised: bool | None = None,
        _data_dir: Path | None = None,
    ) -> list[tuple[object, int]]:
        return self._train_samples if split == "train" else self._test_samples
