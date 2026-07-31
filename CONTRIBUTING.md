# Contributing to dagnam

Thanks for helping improve the Dagnam.AI Python SDK.

This repository currently uses a small-maintainer workflow. Please keep changes
focused, include tests for behavior changes, and avoid unrelated formatting
churn.

## Development Setup

```bash
cd dag-lib
uv sync
```

Run every quality gate the way CI does, with a single command:

```bash
uv run poe check        # ruff + format-check + pyright + lint-imports + pytest
```

Or run the individual gates:

```bash
uv run poe lint          # ruff check
uv run poe format-check  # ruff format --check
uv run poe typecheck     # pyright (strict)
uv run poe lint-imports  # layer-first import contract
uv run poe test          # parallel pytest + terminal coverage
uv run poe test-cov      # parallel pytest + terminal, HTML, JSON, and XML coverage reports
uv run poe audit         # dependency vulnerability audit
uv run poe build         # source distribution and wheel
```

Run `uv run poe check` before pushing.

The `test` and `test-cov` tasks use four pytest-xdist workers with
`--dist=loadfile`. Tests from one file stay together and run sequentially while
independent files run in parallel.

## Coverage Reports

Generate the complete statement and branch coverage report with:

```bash
uv run poe test-cov
```

The command enforces the configured 100% coverage gate and writes:

- Terminal report with missing lines
- `htmlcov/index.html` for the browsable report
- `coverage.json` for machine-readable detail
- `coverage.xml` for CI and coverage services

These generated files are ignored by Git.

## Architecture

`dagnam` is **layer-first** by design (this is the conventional shape for a
client SDK; cf. `openai`, `stripe`):

```
dagnam/
  _core/      # transport, auth, config, client, LRO, exceptions  (base layer)
  data/       # datasets, cache, loaders, adapters, load_dataset   (depends on _core)
  resources/  # high-level resource modules (datasets, hub, ...)   (depends on data/_core)
  cli/        # argparse CLI                                        (top layer)
```

The import direction `cli -> resources -> data -> _core` is enforced by
`import-linter` (`uv run poe lint-imports`, blocking in CI): a layer may import
only the layers below it.

### Public-API contract

`tests/test_public_api.py` is a golden test that pins `dagnam.__all__`, the
documented submodule paths, and the lazy exports. **Keep it green.** Any change
to the public surface must be a deliberate, `CHANGELOG.md`-recorded decision, not
an accident of a refactor.

### Imports

Module-level imports go at the top of the file (ruff `E402`). The SDK
intentionally uses **lazy imports inside functions** in two situations, both
sanctioned: (1) optional framework backends (`torch`, `tensorflow`, `jax`/`flax`,
`torchaudio`, `PIL`, `sseclient`) so a base install stays light — the CI
"base-install isolation" job verifies these never leak into `import dagnam`; and
(2) the CLI, which lazy-imports per-command to keep startup fast.

## Branching

`main` is the trunk and is always releasable. Work on a short-lived topic branch
named `<type>/<slug>` — `feat`, `fix`, `chore`, `docs`, `refactor`, `perf`,
`test`, `sec`, `build`, `ci` or `revert` — and open a pull request:

```bash
git switch -c fix/dataset-cache-lock
git push -u origin fix/dataset-cache-lock
```

Releases are cut by tag: pushing `dagnam/v*` builds, attests and publishes to
PyPI. A `release/X.Y` branch exists only when a published minor needs a patch
that `main` has already moved past. **Fixes land on `main` first**, then get
cherry-picked onto the release branch — a fix that lives only on a release
branch reappears as a regression at the next minor.

## Pre-push Hook

`.githooks/pre-push` refuses direct pushes to `main` (changes arrive by PR so CI
runs first) and refuses force-pushes to or deletion of `main` and `release/*`.
Install it after cloning:

```bash
cp .githooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

It is installed into `.git/hooks/` rather than via `core.hooksPath`, because
setting that would make Git search only that directory and silently disable the
`pre-commit` framework's hook. `git push --no-verify` bypasses it.

## Pull Request Expectations

- Explain the user-visible change and why it is needed.
- Add or update tests for bug fixes and new behavior.
- Update `README.md` or `CHANGELOG.md` when the public API, CLI, packaging,
  compatibility, or security posture changes.
- Keep public APIs backwards-compatible within the current release line unless a
  breaking change is explicitly planned.
- Do not include API keys, credentials, private dataset URLs, generated caches,
  or large binary artifacts.

## Code Style

The project uses Ruff for linting and formatting, Pyright for static checking,
and pytest for tests. Prefer existing module patterns over new abstractions.

## Pre-commit Hooks

The tracked `.pre-commit-config.yaml` is temporarily disabled with `repos: []`
so local Git hooks do not block commits or rewrite files during a commit. Before
re-enabling the commented workflow, make the checks advisory or move automatic
fixes to explicit developer commands.

## Reporting Security Issues

Please do not report vulnerabilities in public issues. Follow
[SECURITY.md](SECURITY.md).
