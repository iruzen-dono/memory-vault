# Contributing to Memory Vault

**Thank you for wanting to build this with me.** Memory Vault started as a solo project — but a protocol is only useful when many people adopt it. Here's how you can help.

## Code of Conduct

Be excellent. This is a small, early project. Every contributor shapes the culture.

## Where to Start

| Your skills | Good first issue |
|-------------|------------------|
| Python (stdlib) | Pack format parsers, CLI features |
| Python + Hermes | Hermes plugin integration, session reader |
| Web (Next.js) | Memory Registry web hub |
| DevOps | CI/CD, release automation, PyPI publishing |
| Docs | Tutorials, examples, integration guides |
| Rust (future) | High-performance pack processing |
| Any | Bug reports, feature requests, tests |

## Project Structure

```
memory-vault/
├── src/memory_vault/
│   ├── core/            # Format: manifest, pack, builder
│   ├── cli/             # Typer CLI
│   ├── formats/         # Hermes-specific readers
│   └── vault/           # Obsidian sync
├── tests/               # Pytest suite
├── docs/                # Architecture, guides
└── .github/             # CI, issue templates
```

## Development Setup

```bash
# Clone & install
git clone https://github.com/iruzen-dono/memory-vault
cd memory-vault
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check src/

# Type-check
mypy src/
```

## Design Principles

1. **Local-first** — everything works offline, no server required
2. **Composable** — each subsystem (profile, memory, skills) is optional in a pack
3. **Human-readable** — JSON and Markdown, not binary blobs
4. **Agent-agnostic** — designed for Hermes first, but the format must work for any agent
5. **No vendor lock** — the format is the standard, not the implementation

## Pull Request Process

1. Open an issue first for non-trivial changes (so we agree on direction)
2. Keep PRs focused — one feature/fix per PR
3. Add tests for new functionality
4. Run `pytest` and `ruff check src/` before requesting review
5. Update docs if you change the format or CLI

## Commit Style

```
verb(scope): description

Examples:
  feat(cli): add --format flag to export command
  fix(pack): handle empty skills directory gracefully
  docs: add quickstart tutorial
  refactor(core): extract manifest validation from pack
```

## Getting Help

- Open a GitHub Discussion
- Tag @iruzen-dono in issues
- Join the Nous Research Discord (`#plugins-skills-and-skins`)

## The Vision

> Memory Vault is the **Git for agent memory**. Not locked to Hermes, not locked to one format, not locked to one backend. A standard that lets any AI agent carry its context across machines, share it with peers, and survive resets.

We're early. Every commit counts.
