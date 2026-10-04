# Release Process

This document is for maintainers publishing `dagnam` to PyPI.

Releases are **automated**: pushing a tag of the form `dagnam/v<version>` (for
example `dagnam/v0.7.0`) triggers [`.github/workflows/release.yml`](.github/workflows/release.yml),
which builds the wheel + sdist, attaches a build-provenance attestation,
publishes to **TestPyPI**, installs from TestPyPI and smoke-imports the package,
and only then publishes to **PyPI**. Publishing uses PyPI **trusted publishing**
(OIDC) through the `testpypi` and `pypi` GitHub environments — there are no API
tokens to manage.

## Platform first

**Deploy the platform before you tag.** The SDK redacts audit rows with the
`dagnam-contracts` version it installs, and the platform re-scans them with its
own; the contract's PII class list and score shape are a wire protocol between
the two. An SDK published ahead of its platform uploads the rows and then stops
every workload at the PII check, for every user, until the platform catches up.

Three things hold that order:

- `pyproject.toml` caps the contract below the next minor
  (`dagnam-contracts>=X.Y.Z,<X.(Y+1)`), so a new contract minor reaches users
  only through an SDK release built against it. Raising the floor is a release
  of its own: do it only once the platform runs that contract.
- The release workflow's first job runs
  `python scripts/check_platform.py https://api.dagnam.ai` and nothing is built
  or published unless it passes. It reads the platform's `GET /health/build`
  and fails unless the `contracts` version reported there is at or above the
  floor in `pyproject.toml` (compared on major and minor). A platform too old
  to report the key, or one that cannot be reached, fails too. The request
  names itself, has a timeout and refuses a redirect to another host. The
  workflow also runs the whole CI workflow on the tagged commit, and `build`
  waits for both; every publishing job waits for `build`, and none can be
  skipped or allowed to fail.
- `dagnam audit run` makes the same comparison before its first upload (with
  `--local-only` too, and from the library's `run_audit`) and stops with
  `platform_too_old` instead of spending anything. A platform a patch ahead of
  the installed contract only warns.

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
   the version bump reflects it. Between releases the work sits under
   `[Unreleased]` on purpose (even when `__version__` is already the next
   number); a dated heading is written only at release time, never ahead of it.
   This command refuses a changelog with entries left under `[Unreleased]`, or
   whose newest dated heading is not the `__version__` being published with a
   real date, and exits non-zero until it is folded:

   ```bash
   python scripts/check_changelog.py
   ```
2. Confirm the platform already runs this release's contract (see
   [Platform first](#platform-first)). The workflow repeats the check as its
   first job; passing it here first saves a release run that stops there:

   ```bash
   python scripts/check_platform.py https://api.dagnam.ai
   ```

3. Run the verification suite:

   ```bash
   uv sync
   uv run poe release-check   # check (lint/format/types/imports/tests) + audit + build
   ```

4. Build and inspect the distribution in a clean `dist/`. `uv build` adds to
   whatever `dist/` already holds, so remove it first -- an artifact left by an
   earlier build would otherwise be checked, smoke-tested or uploaded as this
   release:

   ```bash
   rm -rf dist
   uv run poe build
   python -m twine check dist/*
   ```

   The `scripts/build-wheel.sh` (Linux/macOS) and `scripts/build-wheel.ps1`
   (Windows) helpers clean `dist/` and run `uv build` for you (pass
   `--no-clean` / `-NoClean` to keep an existing `dist/`).

5. Install the wheel in a clean environment and smoke-test import, CLI, and
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

Watch the run under **Actions → Release dagnam to PyPI**. The platform check
runs first: if it fails, deploy the platform, then re-run the failed job (the
tag stays). TestPyPI publish and the install smoke-test run *before* PyPI, so a
broken build is caught before it reaches the real index. If you configure a
required reviewer on the `pypi` environment, the final publish step waits for
manual approval.

### First-time setup (once per index)

Trusted publishing must be configured before the first automated release:

- On **TestPyPI** and **PyPI**, add a *pending publisher* for PyPI project
  `dagnam`, owner `Dagnam-AI`, repository `dag-lib`, workflow file
  `release.yml`, and environment `testpypi` / `pypi` respectively.
- Create matching GitHub environments `testpypi` and `pypi` under
  **Settings → Environments** (optionally add a required reviewer on `pypi`).

### Manual fallback

If you must publish by hand (trusted publishing unavailable), run the platform
check yourself, build into a **fresh directory**, and upload from it with an API
token. Never upload `dist/*` from a working tree: `dist/` is not cleaned between
builds, so the glob can pick up a wheel or sdist an earlier build left behind.

```bash
python scripts/check_platform.py https://api.dagnam.ai
out="$(mktemp -d)"
uv build --out-dir "$out"
python -m twine check "$out"/*
python -m twine upload --repository testpypi "$out"/*
python -m twine upload "$out"/*
```

## Post-release

- Verify the PyPI project page renders the README correctly.
- Verify `pip install dagnam` works in a fresh environment.
- Confirm `CHANGELOG.md` has a fresh empty `[Unreleased]` for the next cycle.
