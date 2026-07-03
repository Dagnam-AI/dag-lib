from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_advertises_python_3_12_plus_support() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    # No upper Python cap — a pure-Python SDK must not lock users off 3.13/3.14+.
    assert project["requires-python"] == ">=3.12"
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]


def test_version_string_matches_across_pyproject_and_package() -> None:
    # The in-tree ``__version__`` is the source-checkout fallback for
    # resolve_version(); it must never drift from the packaged version.
    import dagnam

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert dagnam.__version__ == pyproject["project"]["version"]


def test_base_dependencies_include_numpy_and_pillow() -> None:
    # numpy is imported eagerly by the dataset layer and Pillow by the image
    # loaders, so a plain ``pip install dagnam`` must pull both — otherwise the
    # first ``load_dataset`` dies with a bare ModuleNotFoundError.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = pyproject["project"]["dependencies"]
    names = {req.split(">")[0].split("=")[0].split("<")[0].strip().lower() for req in base}
    assert "numpy" in names
    assert "pillow" in names


def test_ml_extras_have_installable_floors() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    # torchvision backs the PyTorch image loaders (ImageFolder/transforms).
    assert optional_dependencies["pytorch"] == ["torch>=2.4", "torchvision>=0.19"]
    # torchaudio>=2.x decodes via TorchCodec, so torchcodec is a runtime dep (G083).
    assert optional_dependencies["audio"] == [
        "torch>=2.12.1",
        "torchaudio>=2.0",
        "torchcodec>=0.1",
    ]
    # TF 2.12 cannot install on Python 3.12; 2.16 is the first 3.12-capable release.
    assert optional_dependencies["tensorflow"] == ["tensorflow>=2.16"]
    assert optional_dependencies["flax"] == [
        "jax>=0.4",
        "flax>=0.7",
    ]
    assert "torchvision>=0.19" in optional_dependencies["all"]
    assert "tensorflow>=2.16" in optional_dependencies["all"]
    assert "jax>=0.4" in optional_dependencies["all"]
    assert "flax>=0.7" in optional_dependencies["all"]


def test_ci_matrix_uses_python_3_12() -> None:
    workflow = (ROOT / ".github" / "workflows" / "dag-lib-ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.12"]' in workflow
    assert 'python-version: ["3.13", "3.14"]' not in workflow
