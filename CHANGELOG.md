# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

First public release. Published to PyPI as **`hermes-memory-vault`**, because the
name `memory-vault` was already taken there by an unrelated project. The import
package is still `memory_vault` and the CLI is still `memory-vault`.

### Added

- `memory-vault` CLI: `browse`, `list-sessions`, `export`, `info`, `import-pack`,
  `index`, `providers`, `config`.
- Interactive TUI session browser (`--extra tui`).
- `.hermes-memory` context packs: conversation, tool traces, artifacts and a
  handoff brief in a single portable archive.
- Smart session indexing with a configurable worker count.
- Provider management and persisted configuration.
- Webhook entry point (`memory-vault-webhook`).
- CI matrix across Ubuntu and Windows on Python 3.11 / 3.12 / 3.13.
- CI `package` job: builds sdist + wheel, runs `twine check`, installs the wheel
  in a clean venv and smoke-tests both entry points.

### Changed

- Distribution renamed `memory-vault` → `hermes-memory-vault` (PyPI name clash).
- Author metadata normalised to Jules Zhou.
- Licence switched to a PEP 639 expression (`license = "MIT"` +
  `license-files`), and the now-illegal licence classifier removed.

### Notes

- `setuptools>=77` is required for PEP 639 support.
