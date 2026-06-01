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

Useful commands:

```bash
uv run pytest tests/ -v
uv run ruff check
uv run ruff format --check
uv run --with "pyright>=1.1.380" pyright
uv run --with pip-audit pip-audit
```

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
