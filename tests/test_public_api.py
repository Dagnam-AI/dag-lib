"""Golden guard for the dagnam public API.

This is the contract that every refactor phase must keep green: the set of
exported names, the importability of every documented ``dagnam.*`` path, and the
resolution of the lazy exports. A change that fails this test is a public-API
break and must be a deliberate, CHANGELOG'd decision coordinated with consumers
(notably ``mvp-backend``), not an accident of a hygiene refactor.
"""

from __future__ import annotations

import importlib

import dagnam

# The exported public surface (mirrors ``dagnam.__all__``). Kept sorted so a
# diff to this list is an obvious, reviewable change to the public contract.
EXPECTED_ALL = sorted(
    [
        "ChecksumError",
        "DagnamClient",
        "DagnamDataset",
        "LongRunningOperation",
        "TrainingEvent",
        "__version__",
        "account",
        "cancel_training_job",
        "codegen",
        "compute_file_checksum",
        "configure",
        "create_training_job",
        "datasets",
        "delete_training_jobs",
        "deployment_health",
        "deployments",
        "download_checkpoint",
        "evict_lru",
        "get_api_key",
        "get_api_url",
        "get_cache_dir",
        "get_config_value",
        "get_training_job",
        "hub",
        "inference",
        "inference_batch",
        "init",
        "is_cached",
        "list_training_jobs",
        "load_dataset",
        "load_metadata",
        "projects",
        "report_error",
        "report_log",
        "report_metric",
        "report_progress",
        "report_system",
        "save_checksum",
        "save_metadata",
        "stream_training",
        "touch_cache",
        "training_logs",
        "training_metrics",
        "training_metrics_summary",
        "write_training_state",
    ]
)

# Documented, importable submodule paths that callers (and generated code) rely
# on. These are part of the contract even though they are not in ``__all__``.
DOCUMENTED_PATHS = [
    "dagnam.resources.datasets",
    "dagnam.resources.training",
    "dagnam.resources.hub",
    "dagnam.resources.deployments",
    "dagnam.resources.projects",
    "dagnam.resources.inference",
    "dagnam.resources.codegen",
    "dagnam.resources.checkpoints",
    "dagnam.resources.account",
    "dagnam.training",
    "dagnam.data.cache",
    "dagnam.data.dataset",
    "dagnam._core.auth",
    "dagnam._core.client",
    "dagnam._core.config",
    "dagnam._core.exceptions",
    "dagnam._core.lro",
    "dagnam.data.load",
]


def test_public_all_is_stable() -> None:
    assert sorted(dagnam.__all__) == EXPECTED_ALL


def test_every_public_name_resolves() -> None:
    for name in dagnam.__all__:
        assert getattr(dagnam, name) is not None


def test_documented_submodule_paths_import() -> None:
    for path in DOCUMENTED_PATHS:
        assert importlib.import_module(path) is not None


def test_lazy_exports_resolve() -> None:
    from dagnam import DagnamDataset, load_dataset

    assert DagnamDataset is not None
    assert load_dataset is not None
