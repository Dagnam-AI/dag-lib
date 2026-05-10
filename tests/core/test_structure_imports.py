"""Structural import checks for the proposed package layout."""

from __future__ import annotations


def test_new_public_resource_imports_match_top_level_exports():
    import dagnam
    from dagnam.resources import datasets, deployments, hub, projects

    assert dagnam.datasets is datasets
    assert dagnam.deployments is deployments
    assert dagnam.hub is hub
    assert dagnam.projects is projects


def test_legacy_resource_imports_remain_compatible():
    from dagnam.resources import datasets, datasets_upload

    assert datasets_upload.upload is datasets.upload
    assert datasets_upload.upload_from_url is datasets.upload_from_url


def test_loader_modules_use_target_names():
    from dagnam.data.loaders import audio, csv, flax, image_folder, json_array, media, system, tf

    assert csv.create_pytorch_loader is not None
    assert json_array.create_pytorch_loader is csv.create_pytorch_loader
    assert audio.create_pytorch_loader is not None
    assert image_folder.create_pytorch_loader is not None
    assert media.discover_class_folders is not None
    assert system.resolve_system_dataset is not None
    assert tf.create_tensorflow_dataset is not None
    assert flax.create_flax_dataset is not None


def test_core_load_and_package_facades_are_importable():
    from dagnam._core import DagnamClient
    from dagnam._core.load import _is_uuid, load_dataset
    import dagnam.aio
    from dagnam.data.dataset import DagnamDataset

    assert DagnamClient is not None
    assert DagnamDataset is not None
    assert load_dataset is not None
    assert _is_uuid("00000000-0000-0000-0000-000000000000")
    assert dagnam.aio.AsyncDagnamClient is not None
