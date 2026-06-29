# Memory Vault — Portable Context Protocol

> **From agent session to shareable context in one command.**

Memory Vault turns any Hermes session into a portable `.hermes-memory` **context pack** — the conversation, the tool traces, the artifacts created, and a handoff brief that lets another agent pick up exactly where you left off.

```bash
# Launch the interactive TUI browser
memory-vault browse

# List sessions
memory-vault list-sessions

# Export a session into a shareable pack
memory-vault export <session-id> --project-root . -o bot.hermes-memory

# Inspect a pack
memory-vault info bot.hermes-memory

# Import (show handoff + artifacts)
memory-vault import-pack bot.hermes-memory
```

## Features

### 🖥️ Interactive TUI (`memory-vault browse`)

Browse sessions with enriched titles (LLM-generated), search, export, and re-index — all from the terminal.

```
┌────────── Session Browser ───────────┐
│ Config Fix                           │
│ 20260629_06423 · deepseek-v4-flash   │
│ The AI agent fixed config issues…    │
│                                      │
│ LLM: ✅ cloudflare                   │
│ Memory Vault Dev                     │
│ 20260628 · deepseek-v4-flash         │
│ Developed and tested Memory Vault…   │
└──────────────────────────────────────┘
```

Press `i` to re-index a session, `e` to export, `Ctrl+P` to search. The footer shows your active LLM provider with its availability status.

### 📋 Provider Management (`memory-vault providers`)

List all registered LLM providers and whether they're available (creds detected):

```bash
$ memory-vault providers
Provider         Available
──────────────── ────────────
cloudflare       ✅
anthropic        ❌
openai           ❌
```

Persist a provider choice so you don't need env vars:

```bash
memory-vault config set llm.provider cloudflare
memory-vault config get llm.provider   # cloudflare
memory-vault config list               # show all stored config
```

### 🧠 Smart Session Indexing (`memory-vault index`)

Generate descriptive titles and summaries for all your Hermes sessions using an LLM provider. Supports any OpenAI-compatible API and Cloudflare Workers AI out of the box.

```bash
# Index all sessions (auto-detects provider)
memory-vault index

# Index with 4 parallel workers (much faster)
memory-vault index --workers 4

# Use Cloudflare Workers AI
export CLOUDFLARE_ACCOUNT_ID="..."
export CLOUDFLARE_API_TOKEN="..."
memory-vault index --force

# Use any OpenAI-compatible API
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
memory-vault index

# Override the model per run
MEMORY_VAULT_INDEX_MODEL="gpt-4o" memory-vault index
```

### 📦 Context Packs

Shareable, self-contained archives of any Hermes session.

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
├── manifest.json           # Metadata: title, tags, source session, git commit, tool usage
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

### 1. Install

```bash
pip install "memory-vault[tui]"   # with TUI
pip install "memory-vault"        # CLI only (lighter)
```

### 2. Browse sessions

```bash
memory-vault browse
# or just list:
memory-vault list-sessions
```

### 3. Index sessions (optional, for meaningful titles)

Requires an LLM provider. By default tries `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` for Cloudflare Workers AI. Falls back gracefully with template titles.

```bash
# Auto-detect Cloudflare Workers AI
memory-vault index

# Or set any OpenAI-compatible provider
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
memory-vault index

# Only new sessions (skip already-indexed)
memory-vault index --new
```

### 4. Export a session

```bash
memory-vault export 20260620_143021_a1b2c3 \
  --title "Hyperliquid Trading Bot" \
  --tags trading,hyperliquid,defi \
  --project-root /path/to/project \
  --author iruzen
```

### 5. Share or archive

Send the `.hermes-memory` file anywhere. The receiving agent loads it with:

```bash
memory-vault import-pack bot.hermes-memory
```

## Commands

| Command | Description |
|---------|-------------|
| `browse` | Launch interactive TUI (sessions or packs) |
| `list-sessions` | Browse available Hermes sessions |
| `export <id>` | Pack a session into .hermes-memory |
| `info <pack>` | Inspect a pack (metadata, artifacts, tools) |
| `import-pack <pack>` | Show handoff brief, artifacts, and tool traces |
| `list` | List all packs in a directory |
| `search <pattern>` | Regex search across packs |
| `render <pack>` | Render pack as Markdown or HTML |
| `index [--workers N]` | Enrich session titles & summaries via LLM |
| `providers` | List available LLM providers and their status |
| `config set/get/list` | Manage persistent configuration |
| `diff <pack1> <pack2>` | Compare two context packs side-by-side |

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
├── cli/
│   └── main.py           # Typer commands (export, info, import, list, search, render, index, browse)
├── core/
│   ├── __init__.py
│   ├── llm.py           # LLMProvider ABC + CloudflareAI + OpenAICompatibleProvider
│   ├── config.py         # JSON config persistence (llm.provider, etc.)
│   ├── manifest.py       # Manifest dataclass + serialization
│   ├── pack.py           # ContextPack — in-memory representation + tar.gz I/O
│   ├── builder.py        # ContextBuilder — reads Hermes sessions, builds packs
│   ├── narrator.py       # SessionNarrator — compresses conversations via LLM
│   ├── renderer.py       # Export packs as Markdown or HTML
│   ├── session_index.py  # SessionIndex — SQLite cache for LLM-enriched titles
│   └── tui.py            # MemoryVaultTUI — Textual-based interactive browser
```

## License

MIT
