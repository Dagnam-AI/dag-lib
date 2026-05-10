# Changelog

All notable changes to the `dagnam` Python SDK.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Restructured SDK internals around `resources`, `_core.load`, package-based
  client/dataset/CLI modules, and renamed loader modules.
- Added typed-package metadata via `dagnam/py.typed`.
- Relicensed from Unlicense to Apache License 2.0 to align with the
  industry-standard SDK license used by OpenAI, Google, and Mistral, and
  to provide explicit patent + trademark grants for enterprise adoption.
  Added a `NOTICE` attribution file as required by section 4(d) and
  wired `force-include` in hatchling so it ships with every wheel.

### Compatibility

- `mvp-backend` >= 0.5.0, < 0.7.0

## [0.1.0] - 2026-05-10

Initial public release line.

### Compatibility

- `mvp-backend` >= 0.5.0, < 0.7.0
