from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_advertises_python_3_12_support() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["requires-python"] == ">=3.12,<3.13"
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]
    assert "Programming Language :: Python :: 3.13" not in project["classifiers"]
    assert "Programming Language :: Python :: 3.14" not in project["classifiers"]


def test_all_ml_extras_are_available_on_python_3_12() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert optional_dependencies["pytorch"] == ["torch>=2.12.1", "torchvision>=0.27.1"]
    assert optional_dependencies["audio"] == ["torch>=2.12.1", "torchaudio>=2.0"]
    # TF/Flax system-dataset loading goes through tensorflow_datasets, so both
    # extras carry tfds (Flax also needs a TensorFlow runtime for tfds.as_numpy)
    # plus tfds' undeclared importlib_resources and truststore (OS trust store
    # for downloads behind TLS-inspection proxies).
    assert optional_dependencies["tensorflow"] == [
        "tensorflow>=2.12",
        "tensorflow-datasets>=4.9",
        "importlib_resources>=6.0",
        "truststore>=0.9",
    ]
    assert optional_dependencies["flax"] == [
        "jax>=0.4",
        "flax>=0.7",
        "tensorflow-datasets>=4.9",
        "tensorflow>=2.12",
        "importlib_resources>=6.0",
        "truststore>=0.9",
    ]
    assert "torch>=2.12.1" in optional_dependencies["all"]
    assert "torchvision>=0.27.1" in optional_dependencies["all"]
    assert "torchaudio>=2.0" in optional_dependencies["all"]
    assert "tensorflow>=2.12" in optional_dependencies["all"]
    assert "tensorflow-datasets>=4.9" in optional_dependencies["all"]
    assert "importlib_resources>=6.0" in optional_dependencies["all"]
    assert "truststore>=0.9" in optional_dependencies["all"]
    assert "jax>=0.4" in optional_dependencies["all"]
    assert "flax>=0.7" in optional_dependencies["all"]


def test_ci_matrix_uses_python_3_12() -> None:
    workflow = (ROOT / ".github" / "workflows" / "dag-lib-ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.12"]' in workflow
    assert 'python-version: ["3.13", "3.14"]' not in workflow
