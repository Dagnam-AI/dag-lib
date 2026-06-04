"""Structural import checks for the proposed package layout."""

from __future__ import annotations

import pytest


def test_new_public_resource_imports_match_top_level_exports() -> None:
    import dagnam
    from dagnam.resources import datasets, deployments, hub, projects

    assert dagnam.datasets is datasets
    assert dagnam.deployments is deployments
    assert dagnam.hub is hub
    assert dagnam.projects is projects


def test_legacy_resource_imports_remain_compatible() -> None:
    from dagnam.resources import datasets, datasets_upload

    assert datasets_upload.upload is datasets.upload
    assert datasets_upload.upload_from_url is datasets.upload_from_url


def test_loader_modules_use_target_names() -> None:
    from dagnam.data.loaders import audio, csv, flax, image_folder, json_array, media, system, tf

    assert csv.create_pytorch_loader is not None
    assert json_array.create_pytorch_loader is csv.create_pytorch_loader
    assert audio.create_pytorch_loader is not None
    assert image_folder.create_pytorch_loader is not None
    assert media.discover_class_folders is not None
    assert system.resolve_system_dataset is not None
    assert tf.create_tensorflow_dataset is not None
    assert flax.create_flax_dataset is not None


def test_loaders_lazy_getattr_rejects_unknown_name() -> None:
    import dagnam.data.loaders as loaders

    # Known submodules resolve lazily via __getattr__ (PEP 562)...
    assert loaders.csv is not None
    # ...but unknown attributes raise AttributeError rather than ImportError.
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = loaders.does_not_exist


def test_core_load_and_package_facades_are_importable() -> None:
    from dagnam._core import DagnamClient
    import dagnam.aio
    from dagnam.data.dataset import DagnamDataset
    from dagnam.data.load import _is_uuid, load_dataset

    assert DagnamClient is not None
    assert DagnamDataset is not None
    assert load_dataset is not None
    assert _is_uuid("00000000-0000-0000-0000-000000000000")
    assert dagnam.aio.AsyncDagnamClient is not None
