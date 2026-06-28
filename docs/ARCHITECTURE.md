# Architecture

## Concept

Memory Vault is a **Context Protocol** for AI agent sessions. It extracts the full context of a Hermes session — conversation, tool traces, artifacts, decisions — into a portable `.hermes-memory` archive that can be shared, archived, or fed to another agent as a handoff.

```
┌─────────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Hermes sessions.db │ ─────> │  ContextBuilder   │ ─────> │  .hermes-memory   │
│  (SQLite)            │  read  │  (analyse +       │  write │  (tar.gz archive) │
│                     │        │   assemble)       │        │                  │
└─────────────────────┘        └──────────────────┘        └──────────────────┘
```

## Core abstractions

### Manifest (`core/manifest.py`)

Every pack's identity. Contains:

- **Format metadata**: `format_version`, `pack_type` (`"context-pack"`), creation timestamp
- **Identity**: `title`, `description`, `tags`, `author`
- **Source session info**: `source_session_id`, platform, model, duration, message count
- **Contents index**: narrative structure, artifact list, tool usage counts

### ContextPack (`core/pack.py`)

The in-memory representation. Holds all the data that gets serialised to `.hermes-memory`:

- `manifest` — Manifest object
- `narrative_md` — chronological transcript
- `messages` — raw session messages
- `decisions` — key decision points (extracted)
- `artifacts` — dict of `relative_path -> Path` to source files
- `tool_traces` — tool usage summary dict
- `handoff_md` — brief for the receiving agent
- `references_md` — extracted URLs

Serialisation: `write()` stages everything to a temp directory, then compresses with `tar.gz`. `read()` reverses.

### ContextBuilder (`core/builder.py`)

The engine. Reads from Hermes `state.db` (SQLite) and produces a `ContextPack`:

1. **Session query** — `HermesSessionDB` wraps SQLite with read-only queries
2. **Message extraction** — reads all messages (`active=1`) ordered by id
3. **Tool counting** — counts calls from both `tool_name` (tool messages) and `tool_calls` JSON (assistant messages)
4. **Artifact detection** — scans `write_file`/`patch` tool calls, resolves file paths relative to project root
5. **Narrative generation** — formats messages into a readable markdown story
6. **Handoff generation** — extracts the first user message (goal) and last assistant response (result)
7. **Reference extraction** — finds URLs in messages

## File format

### `.hermes-memory` (tar.gz)

```
manifest.json          — JSON. Always first-entry convention.
narrative.md           — Markdown. Chronological session story.
messages.json          — JSON array. Raw messages from the DB.
decisions.json         — JSON array. Key decisions (optional).
artifacts/             — Directory. Copies of created/modified files.
tool-traces.json       — JSON. Tool call counts and summary.
context/
    handoff.md         — Markdown. Brief for agent handoff.
    references.md      — Markdown. Extracted URLs and resources.
```

### manifest.json

```json
{
  "format_version": "1.0.0",
  "pack_type": "context-pack",
  "created_at": "2026-06-28T20:00:00Z",
  "title": "Built Hyperliquid Trading Bot",
  "description": "Full pipeline from research to working bot",
  "tags": ["trading", "hyperliquid"],
  "author": "iruzen",
  "source_session_id": "20260620_143021_a1b2c3",
  "source_platform": "cli",
  "source_model": "claude-sonnet-4",
  "duration_minutes": 45,
  "message_count": 142,
  "narrative": { "chapters": [] },
  "artifacts": {
    "count": 8,
    "files": ["src/bot.py", "src/strategy.py", "docker-compose.yml"]
  },
  "tool_usage": {
    "total_calls": 120,
    "by_tool": {
      "terminal": 40,
      "write_file": 12,
      "read_file": 30
    }
  }
}
```

## Data flow

```
User runs: memory-vault export <session-id>

  1. ContextBuilder.__init__()
     → Detects HERMES_HOME (env var → platform default)
     → Points to state.db

  2. HermesSessionDB.get_session(id)
     → SQL: SELECT * FROM sessions WHERE id = ?

  3. HermesSessionDB.get_messages(id)
     → SQL: SELECT * FROM messages WHERE session_id = ? AND active = 1

  4. ContextBuilder.build_from_session()
     → Counts tool usage across all messages
     → Detects artifact tool calls (write_file/patch)
     → Resolves relative paths (project_root > session.cwd > fp.name)
     → Builds narrative markdown
     → Builds handoff markdown
     → Builds references markdown

  5. ContextPack.write(path)
     → Creates staging temp dir
     → Writes all files in the layout
     → Compresses to tar.gz

  6. Output: .hermes-memory file ready to share
```

## Design decisions

### Why tar.gz and not a directory?

- Single file = one artifact to share, upload, archive
- Compression shrinks large sessions (messages.json can be hundreds of KB)
- Handles deep directory structures for artifacts
- Readable with any `tar` tool (no custom parser needed)

### Why relative paths for artifacts?

Absolute paths would:
- Leak filesystem structure (home directories, usernames)
- Break on different machines
- Collide when joined with temp dirs

We try `project_root` → `session.cwd` → `session.git_repo_root` → filename. The user controls `--project-root` for the best results.

### Why keep raw messages.json?

The narrative is great for humans, but another AI agent benefits from seeing the raw messages — tool call arguments, exact output, reasoning blocks. The JSON preserves every detail.

### Why a handoff brief?

The handoff.md is the **first thing** another agent reads. It answers:
- What was the goal?
- What was achieved?
- What model and platform was used?
- What system prompt guided the work?

This lets the receiving agent continue without asking "what is this?".
