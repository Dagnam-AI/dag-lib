"""Shared test fixtures for the dagnam client library."""

import json
from pathlib import Path

import pytest


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
