# Changelog

All notable changes to `dagnam-record` are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A stdlib-only recorder of `openai` and `anthropic` calls (no dependencies, Python 3.9 and newer,
  no network code). One JSON line per call, in a daily file under `~/.dagnam/traces` (directory
  `0700`, files `0600`): the raw request and response, a provider-neutral normalized copy, the
  time, latency and HTTP status. Headers, keys and `extra_*` arguments are never recorded; media is
  replaced by its size.
- `install()` with the options `window_days` (default 14, recording stops after the window and
  daily files older than it are removed when the recorder is set up), `dir`, `sample`, `no_raw`
  and `redact` (a function, or `"auto"` with the optional `dagnam-record[redact]` extra, which needs
  Python 3.12 or newer). Also `record_call()` for frameworks that bypass the SDKs, `uninstall()`
  and `is_installed()`.
- Recorded: Chat Completions and Responses (`create`, `parse`, `stream()` helpers, streams),
  Messages (`create`, `parse`, `stream()`, the `beta` twins, tool runner requests, streams, and
  Bedrock and Vertex through the same classes), sync and async. A stream is written when its end was
  seen; an abandoned or cut stream is counted as `dropped_partial`. Failed calls are recorded with
  the error's HTTP status, or 599 when there was no HTTP response.
- The kill switch `DAGNAM_RECORD` (`0`, `false`, `off`, `no`, `disable`, `disabled`).
- `python -m dagnam_record status` and `purge` (with `--dry-run`), exit codes 0, 1 and 2.
- The tested SDK ranges (`openai>=1.66,<3`, `anthropic>=0.40,<1`): the version of each patched SDK is
  saved as `sdk_versions` in `.status.json`; outside its range an SDK's calls are written as the
  normalized form only, and the recorder logs one warning. `record_call()` is not narrowed by it.
- It fails open: an error in the recorder is counted, never raised into the program; a full queue
  or a slow disk never blocks a call. `install()` does not raise when an SDK is half-installed (an
  import that fails with something other than `ImportError`): that SDK is skipped and one warning
  is logged.
- Release automation (`release-record.yml`, tag `dagnam-record/v<version>`) and a CI matrix on
  Python 3.9, 3.11 and 3.13 against the lowest and highest supported `openai` and `anthropic`.
