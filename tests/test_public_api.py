"""Golden guard for the dagnam public API.

This is the contract that every refactor phase must keep green: the set of
exported names, the importability of every documented ``dagnam.*`` path, and the
resolution of the lazy exports. A change that fails this test is a public-API
break and must be a deliberate, CHANGELOG'd decision coordinated with downstream
consumers, not an accident of a hygiene refactor.
"""

from __future__ import annotations

import importlib

import dagnam
from dagnam._core import exceptions as _core_exceptions

# Every user-catchable exception, re-exported at the top level so callers can
# write ``except dagnam.APIError`` without reaching into the private
# ``dagnam._core.exceptions`` path.
EXCEPTION_EXPORTS = [
    "AccountLockedError",
    "AccountSuspendedError",
    "APIError",
    "ArchitectureValidationError",
    "ArchitectureVersionNotFoundError",
    "AuthError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "ChecksumError",
    "CodegenError",
    "CodegenValidationError",
    "DagnamError",
    "DatasetNotFoundError",
    "DeploymentNotFoundError",
    "DeploymentStateError",
    "DeploymentValidationError",
    "EmailNotVerifiedError",
    "HubError",
    "HubModelNotFoundError",
    "InvalidURLError",
    "LROFailedError",
    "LROTimeoutError",
    "ModelError",
    "ModelNotFoundError",
    "PayloadTooLargeError",
    "ProjectNotFoundError",
    "QuotaExceededError",
    "ResponseError",
    "StreamError",
    "TaskNotFoundError",
    "TrainingJobNotFoundError",
    "UploadError",
]

# The exported public surface (mirrors ``dagnam.__all__``). Kept sorted so a
# diff to this list is an obvious, reviewable change to the public contract.
EXPECTED_ALL = sorted(
    [
        *EXCEPTION_EXPORTS,
        "AsyncDagnamClient",
        "DagnamClient",
        "DagnamDataset",
        "LongRunningOperation",
        "TrainingEvent",
        "__version__",
        "account",
        "allowed_strategies",
        "cancel_training_job",
        "codegen",
        "compute_file_checksum",
        "configure",
        "create_training_job",
        "datasets",
        "delete_dataset",
        "delete_training_jobs",
        "deployment_health",
        "deployments",
        "download_checkpoint",
        "download_code",
        "download_dag",
        "download_project_thumbnail",
        "enable_debug_logging",
        "estimate_resources",
        "evict_lru",
        "get_api_key",
        "get_api_url",
        "get_cache_dir",
        "get_config_value",
        "get_training_job",
        "hub",
        "inference",
        "inference_batch",
        "inference_schema",
        "inference_stream",
        "init",
        "is_cached",
        "list_training_jobs",
        "load_dataset",
        "load_metadata",
        "models",
        "preview_dataset",
        "projects",
        "report_error",
        "report_log",
        "report_metric",
        "report_progress",
        "report_system",
        "restart",
        "restore_checkpoint",
        "save_checksum",
        "save_metadata",
        "stream_training",
        "studio",
        "touch_cache",
        "training_logs",
        "training_metrics",
        "training_metrics_summary",
        "update_dataset",
        "update_dataset_roles",
        "upload_project_thumbnail",
        "write_training_state",
    ]
)

# Documented, importable submodule paths that callers (and generated code) rely
# on. These are part of the contract even though they are not in ``__all__``.
DOCUMENTED_PATHS = [
    "dagnam.resources.datasets",
    "dagnam.resources.training",
    "dagnam.resources.hub",
    "dagnam.resources.models",
    "dagnam.resources.deployments",
    "dagnam.resources.projects",
    "dagnam.resources.studio",
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
    "dagnam.exceptions",
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


def test_exception_hierarchy_is_exported_at_top_level() -> None:
    # Every catchable exception is reachable as ``dagnam.<Name>`` and is the
    # SAME object as the canonical one in the private module.
    for name in EXCEPTION_EXPORTS:
        exported = getattr(dagnam, name)
        assert exported is getattr(_core_exceptions, name)
        assert issubclass(exported, dagnam.DagnamError)


def test_exceptions_submodule_reexports_full_hierarchy() -> None:
    exceptions_module = importlib.import_module("dagnam.exceptions")
    for name in EXCEPTION_EXPORTS:
        assert getattr(exceptions_module, name) is getattr(_core_exceptions, name)
    assert sorted(exceptions_module.__all__) == sorted(EXCEPTION_EXPORTS)
