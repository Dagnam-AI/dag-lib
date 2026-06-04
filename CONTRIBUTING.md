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
uv run poe test          # pytest
uv run --with pip-audit pip-audit
```

Run `uv run poe check` before pushing.

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
