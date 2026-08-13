"""Shared test fixtures for the dagnam client library.

Importing torchvision here is a load-order workaround, not a convenience
import, and it must stay the FIRST thing this module does.

`import tensorflow` followed by `import torchvision` segfaults the interpreter
(SIGSEGV, exit 139) during torchvision's own import. It is deterministic and
reproduces in two lines:

    python -c "import tensorflow; import torchvision"   # exit 139
    python -c "import torchvision; import tensorflow"   # exit 0

`import torch` first is NOT sufficient — torchvision specifically has to precede
tensorflow. The suite exercises the PyTorch, TensorFlow and Flax loaders in one
process, so whichever happens to import first decides whether the run survives;
that is why the whole suite crashed at collection while any single directory
passed on its own (each shard only ever loaded one of the two).

Doing it here fixes the whole session, because pytest imports the root conftest
before collecting any test module. The library itself deliberately does NOT do
this: `dagnam` loads frameworks lazily (PEP-562) so importing the SDK never
drags torch/tensorflow in, and eagerly importing torchvision to dodge an
upstream ABI clash would trade a real design property for a test-environment
problem. Callers who mix both frameworks in one process need the same ordering;
that is documented for them rather than forced on everyone.
"""

from collections.abc import Callable
import importlib
import json
from pathlib import Path

import pytest

# Imported for its side effect (load order) and nothing else, so it goes through
# importlib rather than a bare `import torchvision` that every linter and type
# checker would correctly flag as unused. Must run before anything pulls in
# tensorflow -- see the module docstring.
try:  # pragma: no cover - depends on whether the pytorch extra is installed
    importlib.import_module("torchvision")
except ImportError:  # pragma: no cover - torch-less installs skip those tests anyway
    pass


@pytest.fixture(autouse=True)
def unforced_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin colour off suite-wide so stderr assertions ignore the caller's shell.

    `dagnam.cli.errors` resolves colour from `FORCE_COLOR` *before* `NO_COLOR`
    and tty detection, so a developer or CI image that exports `FORCE_COLOR`
    wraps every `Error:` in ANSI codes and reddens ~28 plain-substring
    assertions that have nothing to do with their change. Setting `NO_COLOR`
    here would not help — it loses to `FORCE_COLOR`. Deleting the variable is
    what makes the decision hermetic. `tests/cli/test_errors.py` already did
    this locally; this generalizes it to every test that asserts on stderr.
    """
    monkeypatch.delenv("FORCE_COLOR", raising=False)


@pytest.fixture
def cache_dir(tmp_path: Path):
    """Temporary cache directory for dataset storage during tests."""
    d = tmp_path / "dagnam_cache"
    d.mkdir()
    return d


@pytest.fixture
def sample_metadata():
    """Sample metadata dict matching the MetadataResponse shape from the API."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Iris Dataset",
        "format": "csv",
        "dataset_type": "tabular",
        "num_samples": 150,
        "num_classes": 3,
        "feature_schema": {
            "columns": [
                {"name": "sepal_length", "type": "numeric"},
                {"name": "sepal_width", "type": "numeric"},
                {"name": "petal_length", "type": "numeric"},
                {"name": "petal_width", "type": "numeric"},
                {"name": "species", "type": "categorical"},
            ]
        },
        "class_names": ["setosa", "versicolor", "virginica"],
        "checksum": "sha256:abc123def456789",
        "file_size": 4096,
        "filename": "iris.csv",
    }


@pytest.fixture
def sample_csv_data():
    """Sample CSV string content for testing loaders."""
    return (
        "sepal_length,sepal_width,petal_length,petal_width,species\n"
        "5.1,3.5,1.4,0.2,setosa\n"
        "4.9,3.0,1.4,0.2,setosa\n"
        "7.0,3.2,4.7,1.4,versicolor\n"
        "6.4,3.2,4.5,1.5,versicolor\n"
        "6.3,3.3,6.0,2.5,virginica\n"
        "5.8,2.7,5.1,1.9,virginica\n"
    )


@pytest.fixture
def sample_json_data():
    """Sample JSON string content for testing loaders."""
    records = [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "species": "setosa",
        },
        {
            "sepal_length": 4.9,
            "sepal_width": 3.0,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "species": "setosa",
        },
        {
            "sepal_length": 7.0,
            "sepal_width": 3.2,
            "petal_length": 4.7,
            "petal_width": 1.4,
            "species": "versicolor",
        },
        {
            "sepal_length": 6.4,
            "sepal_width": 3.2,
            "petal_length": 4.5,
            "petal_width": 1.5,
            "species": "versicolor",
        },
        {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5,
            "species": "virginica",
        },
        {
            "sepal_length": 5.8,
            "sepal_width": 2.7,
            "petal_length": 5.1,
            "petal_width": 1.9,
            "species": "virginica",
        },
    ]
    return json.dumps(records, indent=2)


@pytest.fixture
def dataset_version_factory() -> Callable[..., dict[str, object]]:
    """Build a dict matching the backend's `DatasetVersion` response model.

    Mirrors the API's `DatasetVersion` schema field-for-field
    (id, dataset_id, version_number, parent_version_id, operation,
    operation_params, file_path, size_bytes, num_samples, content_hash,
    data_format, rows_removed, rows_changed, is_pinned, created_at) so CLI/client
    tests exercise the real wire shape instead of an invented `{"version": ...}`
    payload. Pass keyword overrides to vary individual fields per test.
    """

    def _build(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "id": "5b1f7e2a-0f3d-4a7a-9c2e-1a2b3c4d5e6f",
            "dataset_id": "9d3c1a2b-4e5f-4a6b-8c7d-2e3f4a5b6c7d",
            "version_number": 1,
            "parent_version_id": None,
            "operation": "upload",
            "operation_params": {},
            "file_path": "datasets/ds1/v1/data.csv",
            "size_bytes": 4096,
            "num_samples": 150,
            "content_hash": "sha256:abc123def456",
            "data_format": "csv",
            "rows_removed": None,
            "rows_changed": None,
            "is_pinned": False,
            "created_at": "2026-01-01T00:00:00Z",
        }
        base.update(overrides)
        return base

    return _build
