# Changelog

All notable changes to `dagnam-record` are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- The package scaffold: a stdlib-only distribution with no dependencies, supporting Python 3.9
  and newer.
- `install()`, `record_call()` and `uninstall()`, the options `window_days`, `dir`, `sample`,
  `no_raw` and `redact` (a function, or `"auto"` with the optional `dagnam-record[redact]` extra),
  `is_installed()`, the kill switch `DAGNAM_RECORD` (`0`, `false`, `off`, `no`, `disable`, `disabled`), pruning of daily files older than the window, and the
  commands `python -m dagnam_record status` and `python -m dagnam_record purge` (with `--dry-run`).
- The tested SDK ranges (`openai>=1.66,<3`, `anthropic>=0.40,<1`): the version of each patched SDK is
  saved as `sdk_versions` in `.status.json`; outside its range an SDK's calls are written as the
  normalized form only, and the recorder logs one warning. `record_call()` is not narrowed by it.
- `install()` no longer raises when an SDK is half-installed (an import that fails with something
  other than `ImportError`): that SDK is skipped and one warning is logged.
