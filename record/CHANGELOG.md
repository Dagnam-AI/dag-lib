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
