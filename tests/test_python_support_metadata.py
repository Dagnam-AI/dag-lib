import hashlib
from importlib import resources
from importlib.metadata import requires
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_advertises_python_3_12_plus_support() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    # No upper Python cap — a pure-Python SDK must not lock users off 3.13/3.14+.
    assert project["requires-python"] == ">=3.12"
    # The classifiers name only what CI tests (see test_ci_matrix_uses_python_3_12).
    versions = [
        c for c in project["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
    ]
    assert versions == ["Programming Language :: Python :: 3.12"]


def test_version_is_single_sourced_from_package_init() -> None:
    # The version lives in exactly one place: ``dagnam.__version__`` in
    # ``dagnam/__init__.py``. hatchling derives the built package version from it
    # (``[tool.hatch.version] path``), so ``[project]`` must NOT carry a static
    # ``version`` and must instead declare it dynamic. This guards against a
    # regression that reintroduces a second, drift-prone copy of the version.
    import dagnam

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert "version" not in project, "version must be dynamic, not statically pinned"
    assert "version" in project.get("dynamic", []), "version must be declared dynamic"
    assert pyproject["tool"]["hatch"]["version"]["path"] == "dagnam/__init__.py"

    # The single source is a real, non-empty version string.
    assert isinstance(dagnam.__version__, str)
    assert dagnam.__version__
    assert dagnam.__version__.strip() == dagnam.__version__


def test_base_dependencies_include_numpy_and_pillow() -> None:
    # numpy is imported eagerly by the dataset layer and Pillow by the image
    # loaders, so a plain ``pip install dagnam`` must pull both — otherwise the
    # first ``load_dataset`` dies with a bare ModuleNotFoundError.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = pyproject["project"]["dependencies"]
    names = {req.split(">")[0].split("=")[0].split("<")[0].strip().lower() for req in base}
    assert "numpy" in names
    assert "pillow" in names


def test_installed_contract_requirement_is_capped_below_the_next_minor() -> None:
    # The contract's PII class list and score shape are a wire protocol with the
    # platform, so a new contract minor must never reach an SDK release that was
    # not built against it: an open floor is how a published SDK started
    # resolving a contract its platform did not run yet. Read from the INSTALLED
    # distribution, which is the metadata a user's resolver sees.
    declared = requires("dagnam") or []
    [contract] = [req for req in declared if re.match(r"dagnam[-_.]contracts\b", req, re.I)]
    specifier = contract.partition(";")[0]
    bounds = dict(re.findall(r"(>=|<=|==|~=|!=|<|>)\s*([0-9][0-9.]*)", specifier))

    assert set(bounds) == {">=", "<"}, f"dagnam-contracts must be a capped range, not {contract}"
    major, minor = (int(part) for part in bounds[">="].split(".")[:2])
    cap = [int(part) for part in bounds["<"].split(".")]
    assert cap[:2] == [major, minor + 1], f"{contract} must stop at the next minor"
    assert not any(cap[2:]), f"{contract} must stop at the next minor"


def test_project_urls_point_at_the_public_repository_and_at_files_it_has() -> None:
    # They are what PyPI links from the project page. They named a repository
    # that does not exist, with a `dag-lib/` path prefix this repository has
    # never had, so the changelog, issue and security links were all dead.
    urls = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["urls"]
    repository = "https://github.com/Dagnam-AI/dag-lib"

    assert urls["Repository"] == repository
    assert urls["Issues"] == f"{repository}/issues"
    for name in ("Changelog", "Security"):
        prefix = f"{repository}/blob/main/"
        assert urls[name].startswith(prefix), urls[name]
        assert (ROOT / urls[name].removeprefix(prefix)).is_file(), urls[name]


def test_ml_extras_have_installable_floors() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    # torchvision backs the PyTorch image loaders (ImageFolder/transforms).
    # torch>=2.13.0 is the validated security floor shared with audio/all
    # (2.12.1 and below carry GHSA-rrmf-rvhw-rf47); keep this in step with
    # pyproject.
    assert optional_dependencies["pytorch"] == ["torch>=2.13.0", "torchvision>=0.19"]
    # Audio decoding uses SoundFile so uploaded WAV/FLAC/MP3 data does not depend
    # on a machine-global FFmpeg shared-library installation. Torchaudio remains
    # required for generated MFCC/Mel transforms, but TorchCodec is not a loader
    # dependency.
    assert optional_dependencies["audio"] == [
        "torch>=2.13.0",
        "torchaudio>=2.0",
        "soundfile>=0.13",
    ]
    # TF 2.12 cannot install on Python 3.12; 2.16 is the first 3.12-capable release.
    # keras>=3.15.0 is a security floor on TensorFlow's own dependency (TF only
    # asks for keras>=3.12.0): 3.15.0 is the first release clearing the six 2026
    # model-loading advisories (GHSA-5gwj-m78q-7pq3, GHSA-v2w2-w228-c444,
    # GHSA-m8wh-29wm-52mv, GHSA-26c4-7vv6-867j, GHSA-gh82-f9x8-5frx,
    # GHSA-58hv-7753-xmfq). Keep it in step with pyproject and the `all` extra.
    assert optional_dependencies["tensorflow"] == ["tensorflow>=2.16", "keras>=3.15.0"]
    assert optional_dependencies["flax"] == [
        "jax>=0.4",
        "flax>=0.7",
    ]
    assert "torchvision>=0.19" in optional_dependencies["all"]
    assert "soundfile>=0.13" in optional_dependencies["all"]
    assert not any(req.startswith("torchcodec") for req in optional_dependencies["all"])
    assert "tensorflow>=2.16" in optional_dependencies["all"]
    assert "keras>=3.15.0" in optional_dependencies["all"]
    assert "jax>=0.4" in optional_dependencies["all"]
    assert "flax>=0.7" in optional_dependencies["all"]


def test_ci_matrix_uses_python_3_12() -> None:
    workflow = (ROOT / ".github" / "workflows" / "dag-lib-ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.12"]' in workflow
    assert 'python-version: ["3.13", "3.14"]' not in workflow


def test_the_distribution_ships_the_token_estimates_tables() -> None:
    # The estimate prices a word by whether the student's vocabulary holds it, a character, a han
    # word or a run of symbols by its tokens, and a repeated unit by whether it is fragile, from
    # three tables that are package data. They
    # are read through importlib.resources, so this holds for an installed wheel, an editable
    # install and a zip alike; the build copies every file of the ``dagnam`` package, which the
    # wheel target's ``packages`` names.
    package = resources.files("dagnam.audit")
    tables = {
        "student_words.z": (
            94_338,
            "e762f4642356db255f9c1a9cbe76eeda980e15de089c17332120ca7d83bdf981",
        ),
        "student_chars.z": (
            94_857,
            "71b57243ccedcf3a94b6c72e336fbe96f225a3579c484beafe8816b4b6b5ffad",
        ),
        "student_fragile.z": (
            2_018,
            "b21b4b02a53e449c589c2f920b0d834444c6aade5200b500ddbfaf0ec4b2dc60",
        ),
    }
    for name, (length, digest) in tables.items():
        table = package.joinpath(name).read_bytes()
        assert len(table) == length
        assert hashlib.sha256(table).hexdigest() == digest
    for retired in ("student_words.bin", "student_chars.bin", "zh_markers.json"):
        assert not package.joinpath(retired).is_file()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "dagnam" in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "dagnam" in pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
