# Memory Vault — Portable Context Protocol

> **From agent session to shareable context in one command.**

Memory Vault turns any Hermes session into a portable `.hermes-memory` **context pack** — the conversation, the tool traces, the artifacts created, and a handoff brief that lets another agent pick up exactly where you left off.

```bash
# List sessions
memory-vault list-sessions

# Export a session into a shareable pack
memory-vault export <session-id> --project-root . -o bot.hermes-memory

# Inspect a pack
memory-vault info bot.hermes-memory

# Import (show handoff + artifacts)
memory-vault import-pack bot.hermes-memory
```

## Why context packs?

AI agent sessions produce **context** — conversations, decisions, files, tool calls — but it's trapped inside the agent's database. Memory Vault extracts it into a structured, portable format that's:

- **Shareable** — send a `.hermes-memory` file to a teammate or another agent
- **Restartable** — the handoff brief tells the receiving agent what was done and how
- **Auditable** — full transcript + tool traces + artifact copies in one file
- **Self-contained** — includes everything needed to understand the session

## The format

A `.hermes-memory` file is a **tar.gz archive** with a clean layout:

```
bot.hermes-memory/
├── manifest.json           # Metadata: title, tags, source session, tool usage
├── narrative.md            # Chronological story of the session
├── messages.json           # Raw session messages (for deep reference)
├── decisions.json          # Key decisions extracted (AI-generated)
├── artifacts/              # Files created or modified during the session
│   ├── src/bot.py
│   └── ...
├── tool-traces.json        # Tool usage summary (calls per tool)
└── context/
    ├── handoff.md           # Brief for the receiving agent
    └── references.md        # URLs and resources used in the session
```

## Quick start

### 1. Browse sessions

```bash
memory-vault list-sessions
# 📋 Recent sessions
#
#   [20260620_143021_a1b2…] Built Hyperliquid Trading Bot
#            cli · claude-sonnet-4 · 142 messages
```

### 2. Export a session

```bash
memory-vault export 20260620_143021_a1b2c3 \
  --title "Hyperliquid Trading Bot" \
  --tags trading,hyperliquid,defi \
  --project-root /path/to/project \
  --author iruzen
```

### 3. Share or archive

Send the `.hermes-memory` file anywhere. The receiving agent loads it with:

```bash
memory-vault import-pack bot.hermes-memory
```

This shows the handoff brief, artifact list, and tool usage statistics — everything needed to continue the work.

## Commands

| Command | Description |
|---------|-------------|
| `list-sessions` | Browse available Hermes sessions |
| `export <id>` | Pack a session into .hermes-memory |
| `info <pack>` | Inspect a pack (metadata, artifacts, tools) |
| `import-pack <pack>` | Show handoff brief, artifacts, and tool traces |
| `list` | List all packs in a directory |

## Use cases

- **Handoff between agents**: finish a session on desktop, continue on Telegram
- **Project snapshots**: capture the full context of building a feature
- **Debugging archives**: save the session that led to a bug
- **Knowledge base**: build a library of past agent actions
- **Onboarding**: share how a complex task was done

## How it works

Memory Vault reads directly from the Hermes sessions database (`state.db`). It:

1. Queries the session metadata (title, model, platform, duration)
2. Reads all messages (user prompts, assistant responses, tool calls)
3. Builds a chronological narrative from the conversation
4. Detects artifacts by scanning `write_file` and `patch` tool calls
5. Counts tool usage for the summary
6. Generates a handoff brief with the original goal and key results
7. Packages everything into a compressed `.hermes-memory` archive

## Project structure

```
src/memory_vault/
├── __init__.py           # Version + public exports
├── __main__.py           # CLI entry point
├── cli/main.py           # Typer commands (export, info, import, list)
├── core/
│   ├── __init__.py
│   ├── manifest.py       # Manifest dataclass + serialization
│   ├── pack.py           # ContextPack — in-memory representation + tar.gz I/O
│   └── builder.py        # ContextBuilder — reads Hermes sessions, builds packs
```

## License

MIT
