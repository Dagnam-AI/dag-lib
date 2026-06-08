# Release Process

This document is for maintainers publishing `dagnam` to PyPI.

## Pre-release Checklist

1. Confirm `pyproject.toml`, `dagnam/__init__.py`, and `CHANGELOG.md` agree on
   the release version.
2. Confirm `README.md` examples match the current public API.
3. Run the verification suite:

   ```bash
   uv sync
   uv run poe release-check
   ```

4. Build and inspect the distribution:

   ```bash
   Remove-Item -Recurse -Force dist
   uv run poe build
   python -m twine check dist/*
   ```

   The clean-and-build pair is wrapped by a helper so you don't have to
   remember the flags:

   ```bash
   ./scripts/build-wheel.ps1   # Windows / PowerShell
   ./scripts/build-wheel.sh    # Linux / macOS
   ```

   Both clean `dist/` (pass `-NoClean` / `--no-clean` to keep it) and run
   `uv build`, leaving `dist/dagnam-<version>-py3-none-any.whl` plus the sdist.
   Run `python -m twine check dist/*` afterward as above.

   The same script feeds **local training tests**: with
   `DAGNAM_PACKAGE_SOURCE=wheelhouse` and `DAGNAM_LOCAL_PATH=<dag-lib>` set in
   `mvp-backend`, the training pipeline installs `dagnam` from this `dist/` via
   `uv pip install --find-links`, so a real training job exercises the exact
   wheel you're about to publish.

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

- Create and push a git tag matching the package version, for example `v0.5.0`.
- Verify the PyPI project page renders the README correctly.
- Verify `pip install dagnam` works in a fresh environment.
- Start the next `CHANGELOG.md` section under `Unreleased`.
