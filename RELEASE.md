# Release Process

This document is for maintainers publishing `dagnam` to PyPI.

Releases are **automated**: pushing a tag of the form `dagnam/v<version>` (for
example `dagnam/v0.7.0`) triggers [`.github/workflows/release.yml`](.github/workflows/release.yml),
which builds the wheel + sdist, attaches a build-provenance attestation,
publishes to **TestPyPI**, installs from TestPyPI and smoke-imports the package,
and only then publishes to **PyPI**. Publishing uses PyPI **trusted publishing**
(OIDC) through the `testpypi` and `pypi` GitHub environments — there are no API
tokens to manage.

## Version bump (single source of truth)

The version lives in exactly one place: `__version__` in `dagnam/__init__.py`.
hatchling derives the built package version from it (`[tool.hatch.version]`), so
there is no second copy in `pyproject.toml` to keep in sync.

1. Bump `__version__` in `dagnam/__init__.py`.
2. Fold the `CHANGELOG.md` `[Unreleased]` entries into a dated
   `## [X.Y.Z] - YYYY-MM-DD` heading and leave a fresh empty `[Unreleased]`.
3. Confirm `README.md` examples and the compatibility table match the release.

## Pre-release checklist (run locally)

1. Confirm the `CHANGELOG.md` `[Unreleased]` section is **empty** — every entry
   must be folded into the dated release heading being published. A populated
   `[Unreleased]` at publish time means changes (often breaking) are shipping
   undocumented under the version; fold them, and if any are breaking, confirm
   the version bump reflects it.
2. Run the verification suite:

   ```bash
   uv sync
   uv run poe release-check   # check (lint/format/types/imports/tests) + audit + build
   ```

3. Build and inspect the distribution in a clean `dist/`:

   ```bash
   uv run poe build
   python -m twine check dist/*
   ```

   The `scripts/build-wheel.sh` (Linux/macOS) and `scripts/build-wheel.ps1`
   (Windows) helpers clean `dist/` and run `uv build` for you (pass
   `--no-clean` / `-NoClean` to keep an existing `dist/`).

4. Install the wheel in a clean environment and smoke-test import, CLI, and
   version:

   ```bash
   python -m venv .release-smoke
   .release-smoke/bin/pip install --upgrade pip
   .release-smoke/bin/pip install dist/dagnam-*.whl
   .release-smoke/bin/python -c "import dagnam; print(dagnam.__version__)"
   .release-smoke/bin/dagnam --help
   ```

## Publish (automated)

Push the release tag; the workflow does the rest:

```bash
git checkout main && git pull
git tag dagnam/v0.7.0
git push origin dagnam/v0.7.0
```

Watch the run under **Actions → Release dagnam to PyPI**. TestPyPI publish and
the install smoke-test run *before* PyPI, so a broken build is caught before it
reaches the real index. If you configure a required reviewer on the `pypi`
environment, the final publish step waits for manual approval.

### First-time setup (once per index)

Trusted publishing must be configured before the first automated release:

- On **TestPyPI** and **PyPI**, add a *pending publisher* for PyPI project
  `dagnam`, owner `Dagnam-AI`, repository `dag-lib`, workflow file
  `release.yml`, and environment `testpypi` / `pypi` respectively.
- Create matching GitHub environments `testpypi` and `pypi` under
  **Settings → Environments** (optionally add a required reviewer on `pypi`).

### Manual fallback

If you must publish by hand (trusted publishing unavailable), build as above and
upload with an API token:

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

## Post-release

- Verify the PyPI project page renders the README correctly.
- Verify `pip install dagnam` works in a fresh environment.
- Confirm `CHANGELOG.md` has a fresh empty `[Unreleased]` for the next cycle.
