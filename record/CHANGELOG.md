# Changelog

All notable changes to `dagnam-record` are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A stdlib-only recorder of `openai` and `anthropic` calls (no dependencies, Python 3.9 and newer,
  no network code). One JSON line per call, in a daily file under `~/.dagnam/traces` (directory
  `0700`, files `0600`): the raw request and response, a provider-neutral normalized copy, the
  time, latency and HTTP status. Headers, keys and `extra_*` arguments are never recorded; media is
  replaced by its size, before `redact` runs; the credentials of tool definitions (`authorization`,
  `headers`, token-like keys and URL secrets of OpenAI `mcp` tools and Anthropic `mcp_servers`) are
  replaced by `<redacted>`.
- `install()` with the options `window_days` (default 14, recording stops after the window and
  daily files older than it are removed when the recorder is set up), `dir`, `sample`, `no_raw`
  and `redact` (a function, or `"auto"` with the optional `dagnam-record[redact]` extra, which needs
  `dagnam-contracts` 0.4.1 up to below 0.6 and so Python 3.12 or newer). Also `record_call()` for
  frameworks that bypass the SDKs, `uninstall()` and `is_installed()`.
- Recorded: Chat Completions and Responses (`create`, `parse`, `stream()` helpers, streams),
  Messages (`create`, `parse`, `stream()`, the `beta` twins, tool runner requests, streams, and
  Bedrock and Vertex through the same classes), sync and async. A stream is written when its end was
  seen; an abandoned or cut stream is counted as `dropped_partial`. Failed calls are recorded with
  the error's HTTP status, or 599 when there was no HTTP response.
- The kill switch `DAGNAM_RECORD` (`0`, `false`, `off`, `no`, `disable`, `disabled`), read by
  `install()` and again on every call, and a `.disabled` file in the trace directory that stops a
  running process from outside. After the recording window has ended `install()` patches nothing
  and a call does no work (it is counted as `dropped_expired`); restarts after the end no longer
  delete the files that wait for the upload.
- `python -m dagnam_record status` and `purge` (with `--dry-run`), exit codes 0, 1 and 2.
- The tested SDK ranges (`openai>=1.66,<3`, `anthropic>=0.40,<1`): the version of each patched SDK is
  saved as `sdk_versions` in `.status.json`; outside its range an SDK's calls are written as the
  normalized form only, and the recorder logs one warning. `record_call()` is not narrowed by it.
- Limits that keep the recorder from harming its host: at most 32 MiB of text queued for the disk and
  1 GiB of data files per directory (over either, lines are counted as `dropped_full`); a stream
  with more text than a line can hold stops being kept; a line over 1,000,000 bytes first drops its
  normalized copy, so prompts of up to about 1 MB are recorded; an existing directory open to group
  or others is tightened only when it is empty or already holds recorder files, otherwise it is
  refused; a `~` that cannot be expanded raises `ValueError` instead of making a folder named `~`;
  the unicode line separators U+2028, U+2029 and U+0085 are escaped so a line stays one line.
- `uninstall()` leaves a patch that another library put on top of ours in place. A writer thread
  that cannot be started is logged and counted instead of raising out of `install()`, and `close()`
  at exit no longer saves from the calling thread when the writer is stuck on a hung disk.
- It fails open: an error in the recorder is counted, never raised into the program; a full queue
  or a slow disk never blocks a call. `install()` does not raise when an SDK is half-installed (an
  import that fails with something other than `ImportError`): that SDK is skipped and one warning
  is logged.
- Release automation (`release-record.yml`, tag `dagnam-record/v<version>`) and a CI matrix on
  Python 3.9, 3.11 and 3.13 against the lowest and highest supported `openai` and `anthropic`.
