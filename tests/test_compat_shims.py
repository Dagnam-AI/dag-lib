"""Coverage for the ``from X import *`` compatibility wrappers.

These modules forward old import paths to their new homes. Importing each
one executes the forwarding line, which is otherwise unreachable from the
rest of the test suite.
"""

from __future__ import annotations

import importlib

import pytest

SHIMS_ALWAYS = [
    "dagnam._core._common",
    "dagnam._core._resolver",
    "dagnam._core._sse",
    "dagnam.data.loaders.csv_loader",
    "dagnam.data.loaders.json_loader",
    "dagnam.data.loaders.media_utils",
    "dagnam.data.loaders.system_loader",
    "dagnam.resources.datasets_upload",
    "dagnam.services",
    "dagnam.services.checkpoints",
    "dagnam.services.codegen",
    "dagnam.services.datasets_upload",
    "dagnam.services.deployments",
    "dagnam.services.hub",
    "dagnam.services.inference",
    "dagnam.services.projects",
    "dagnam.services.training",
]

SHIMS_OPTIONAL_DEPS = [
    "dagnam.data.loaders.audio_loader",
    "dagnam.data.loaders.flax_loader",
    "dagnam.data.loaders.image_folder_loader",
    "dagnam.data.loaders.tf_loader",
]


@pytest.mark.parametrize("modname", SHIMS_ALWAYS)
def test_shim_imports(modname: str) -> None:
    mod = importlib.import_module(modname)
    assert mod is not None


@pytest.mark.parametrize("modname", SHIMS_OPTIONAL_DEPS)
def test_optional_shim_imports(modname: str) -> None:
    try:
        mod = importlib.import_module(modname)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {modname}: {exc}")
    else:
        assert mod is not None
