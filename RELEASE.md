# Release Process

This document is for maintainers publishing `dagnam` to PyPI.

## Pre-release Checklist

1. Confirm `pyproject.toml`, `dagnam/__init__.py`, and `CHANGELOG.md` agree on
   the release version.
2. Confirm `README.md` examples match the current public API.
3. Run the verification suite:

   ```bash
   uv sync
   uv run pytest tests/ -v
   uv run ruff check
   uv run ruff format --check
   uv run --with "pyright>=1.1.380" pyright
   uv run --with pip-audit pip-audit
   ```

4. Build and inspect the distribution:

   ```bash
   Remove-Item -Recurse -Force dist
   uv build
   python -m twine check dist/*
   ```

5. Install the wheel in a clean environment and smoke-test import, CLI help, and
   metadata:

   ```bash
   python -m venv .release-smoke
   .release-smoke\Scripts\python -m pip install --upgrade pip
   .release-smoke\Scripts\python -m pip install dist\dagnam-*.whl
   .release-smoke\Scripts\python -c "import dagnam; print(dagnam.__version__)"
   .release-smoke\Scripts\dagnam --help
   ```

## Publish

Publish to TestPyPI first when possible:

```bash
python -m twine upload --repository testpypi dist/*
```

Then publish to PyPI:

```bash
python -m twine upload dist/*
```

## Post-release

- Create and push a git tag matching the package version, for example `v0.1.0`.
- Verify the PyPI project page renders the README correctly.
- Verify `pip install dagnam` works in a fresh environment.
- Start the next `CHANGELOG.md` section under `Unreleased`.
