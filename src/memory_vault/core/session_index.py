"""Session Index — local cache of LLM-enriched session titles and summaries.

Scans Hermes sessions, analyzes content via Cloudflare Workers AI,
and stores custom titles + summaries so the TUI shows accurate,
descriptive names instead of auto-generated ones.

Model selection (priority order):
  1. `model` parameter passed explicitly
  2. MEMORY_VAULT_INDEX_MODEL env var
  3. narrator's fast model (llama-3.3-70b by default)

Schema:
    session_id  TEXT PRIMARY KEY
    title       TEXT          — LLM-generated descriptive title
    summary     TEXT          — 1-3 sentence summary of what happened
    model_used  TEXT          — model that generated the title/summary
    indexed_at  REAL          — Unix timestamp
    msg_count   INTEGER       — message count at time of indexing
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from memory_vault.core.narrator import CloudflareAI, _FAST_MODEL, _DEEP_MODEL


# ── Default model selection ────────────────────────────────────────

_DEFAULT_INDEX_MODEL = os.environ.get("MEMORY_VAULT_INDEX_MODEL", _FAST_MODEL)
"""Model used for title+summary generation. Override via env var."""


# ── LLM prompt for title+summary generation ────────────────────────

TITLE_PROMPT = """\
You are analyzing an AI agent conversation session. Based on the transcript excerpt below, produce:

1. A SHORT, descriptive title (max 60 characters, no quotes, just the title)
2. A 1-2 sentence summary of what was accomplished

Format your response EXACTLY like this (no markdown, no extra text):

TITLE: <the title>
SUMMARY: <the summary>

Session metadata:
- Model: {model}
- Platform: {platform}
- Messages: {msg_count}
- Primary tools used: {tools}

## Transcript start

{transcript}

## Transcript end

Remember: TITLE: on first line, SUMMARY: on second line. Max 60 chars for title."""


# ── Indexer ─────────────────────────────────────────────────────────


class SessionIndex:
    """Local SQLite cache of enriched session titles and summaries.

    Usage:
        idx = SessionIndex()
        idx.index_all(builder)        # one-shot: scan everything
        idx.index_new(builder)        # incremental: only unindexed
        entry = idx.get(session_id)   # read cached entry
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else self._default_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ai = CloudflareAI()
        self._init_db()
        self.model = os.environ.get("MEMORY_VAULT_INDEX_MODEL", _FAST_MODEL)

    @staticmethod
    def _detect_hermes_home() -> Path:
        """Minimal Hermes home detection (mirrors ContextBuilder._detect_hermes_home)."""
        env = os.environ.get("HERMES_HOME")
        if env:
            return Path(env)
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "")
            if local_appdata:
                return Path(local_appdata) / "hermes"
            return Path.home() / "AppData" / "Local" / "hermes"
        return Path.home() / ".hermes"

    def _default_path(self) -> Path:
        """Store index next to Hermes state DB so it's profile-aware."""
        return self._detect_hermes_home() / "memory-vault-index.db"

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_index (
                    session_id  TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT '',
                    summary     TEXT NOT NULL DEFAULT '',
                    model_used  TEXT NOT NULL DEFAULT '',
                    indexed_at  REAL NOT NULL,
                    msg_count   INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def set_model(self, model: str) -> None:
        """Override the model used for indexing."""
        self.model = model

    # -- Read API ----------------------------------------------------

    def get(self, session_id: str) -> dict | None:
        """Get indexed entry for a session, or None."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT * FROM session_index WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                keys = ["session_id", "title", "summary", "model_used",
                        "indexed_at", "msg_count"]
                return dict(zip(keys, row))
            return None
        finally:
            conn.close()

    def list_indexed(self, limit: int = 50) -> list[dict]:
        """List all indexed sessions, newest first."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT * FROM session_index ORDER BY indexed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            keys = ["session_id", "title", "summary", "model_used",
                    "indexed_at", "msg_count"]
            return [dict(zip(keys, r)) for r in rows]
        finally:
            conn.close()

    def count(self) -> int:
        """Number of indexed sessions."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            return conn.execute("SELECT COUNT(*) FROM session_index").fetchone()[0]
        finally:
            conn.close()

    def summary_stats(self) -> dict:
        """Return summary statistics about the index."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            total = conn.execute("SELECT COUNT(*) FROM session_index").fetchone()[0]
            with_narrative = conn.execute(
                "SELECT COUNT(*) FROM session_index WHERE summary != ''"
            ).fetchone()[0]
            return {"total": total, "with_summary": with_narrative}
        finally:
            conn.close()

    # -- Write API ---------------------------------------------------

    def _upsert(self, session_id: str, title: str, summary: str,
                model_used: str, msg_count: int) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """INSERT OR REPLACE INTO session_index
                   (session_id, title, summary, model_used, indexed_at, msg_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, title, summary, model_used, time.time(), msg_count),
            )
            conn.commit()
        finally:
            conn.close()

    # -- Indexing logic ----------------------------------------------

    def _build_transcript_sample(self, messages: list[dict]) -> str:
        """Build a compact transcript sample suitable for the LLM prompt."""
        lines = []
        for i, msg in enumerate(messages[:80]):  # cap at 80 msgs
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:400]
            tool_name = msg.get("tool_name", "")
            tool_calls = msg.get("tool_calls")

            if role == "user" and content:
                lines.append(f"User: {content}")
            elif role == "assistant":
                if content:
                    lines.append(f"Assistant: {content[:200]}")
                if tool_calls:
                    try:
                        tc = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                        for t in (tc if isinstance(tc, list) else [tc]):
                            name = t.get("name", t.get("function", {}).get("name", "tool"))
                            lines.append(f"  [tool: {name}]")
                    except (json.JSONDecodeError, TypeError):
                        pass
            elif role == "tool" and tool_name:
                short = (content or "")[:120]
                lines.append(f"  [result: {tool_name}] {short}")

        return "\n".join(lines[:100])  # hard cap

    def _extract_title_summary(self, response: str) -> tuple[str, str]:
        """Parse 'TITLE: ...' and 'SUMMARY: ...' from LLM response."""
        title = ""
        summary = ""
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("TITLE:"):
                title = line[len("TITLE:"):].strip().strip('"').strip("'")
            elif line.upper().startswith("SUMMARY:"):
                summary = line[len("SUMMARY:"):].strip().strip('"').strip("'")

        # Fallback: if LLM didn't follow format, use first meaningful line
        if not title:
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("```"):
                    title = line[:60]
                    break

        title = title[:60]
        summary = summary[:500]

        return title, summary

    def _llm_title_summary(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 512,
    ) -> str | None:
        """Call Cloudflare AI with a custom model, bypassing summarize()."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise session labeler. "
                           "Respond with exactly TITLE: and SUMMARY: lines.",
            },
            {"role": "user", "content": prompt},
        ]
        result = self._ai._call(model, messages, max_tokens=max_tokens, temperature=0.3)
        if result:
            return result.get("response", "")
        return None

    def index_session(
        self,
        session: dict,
        messages: list[dict],
        force: bool = False,
        model: str | None = None,
    ) -> dict | None:
        """Analyze a single session and store enriched title+summary.

        Args:
            session: Session metadata dict (must have 'id', 'title', etc.)
            messages: Full message list for the session.
            force: Re-index even if already indexed.
            model: Model to use for generation. Falls back to
                   MEMORY_VAULT_INDEX_MODEL env var, then default fast model.

        Returns:
            The indexed entry dict, or None if skipped.
        """
        session_id = session.get("id", "")
        if not session_id:
            return None

        msg_count = len(messages)

        # Check if already indexed and unchanged
        if not force:
            existing = self.get(session_id)
            if existing and existing.get("msg_count", 0) >= msg_count:
                return existing

        # Resolve model: explicit > env var > default
        active_model = model or self.model or _FAST_MODEL

        # Build transcript sample
        transcript = self._build_transcript_sample(messages)

        # Gather tool names
        tools_seen = set()
        for msg in messages[:80]:
            tname = msg.get("tool_name", "")
            if tname:
                tools_seen.add(tname)
            tc_raw = msg.get("tool_calls")
            if tc_raw:
                try:
                    tc = json.loads(tc_raw) if isinstance(tc_raw, str) else tc_raw
                    for t in (tc if isinstance(tc, list) else [tc]):
                        name = t.get("name", t.get("function", {}).get("name", ""))
                        if name:
                            tools_seen.add(name)
                except (json.JSONDecodeError, TypeError):
                    pass

        tool_str = ", ".join(sorted(tools_seen)[:8]) or "none"

        # Try LLM path
        title = ""
        summary = ""
        model_used = "template"

        if self._ai.available():
            prompt = TITLE_PROMPT.format(
                model=session.get("model", "?"),
                platform=session.get("source", "?"),
                msg_count=msg_count,
                tools=tool_str,
                transcript=transcript,
            )
            response = self._llm_title_summary(prompt, model=active_model)
            if response:
                title, summary = self._extract_title_summary(response)
                model_used = active_model

        # Fallback: use session title + first message as summary
        if not title:
            title = session.get("title", "") or "(untitled)"
        if not summary:
            for msg in messages:
                if msg.get("role") == "user":
                    content = (msg.get("content") or "")[:200]
                    if content:
                        summary = content
                        break
            if not summary:
                summary = f"{msg_count} messages · tools: {tool_str}"

        title = title[:60]
        summary = summary[:500]

        self._upsert(session_id, title, summary, model_used, msg_count)

        return {
            "session_id": session_id,
            "title": title,
            "summary": summary,
            "model_used": model_used,
            "indexed_at": time.time(),
            "msg_count": msg_count,
        }

    def index_all(
        self,
        builder: "ContextBuilder",  # noqa: F821
        force: bool = False,
        model: str | None = None,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> dict:
        """Index all sessions from the Hermes DB.

        Args:
            builder: ContextBuilder instance for reading sessions.
            force: Re-index even if already indexed.
            model: Model override for LLM generation.
            progress_callback: fn(current, total, session_id, status)
                               where status is "indexing", "skipped", "done", "error".

        Returns:
            Stats dict: {"total": N, "indexed": N, "skipped": N, "errors": N}
        """
        from memory_vault.core.builder import HermesSessionDB

        if not builder.db_path.exists():
            return {"total": 0, "indexed": 0, "skipped": 0, "errors": 0,
                    "error": f"DB not found at {builder.db_path}"}

        db = HermesSessionDB(builder.db_path)
        sessions = db.list_sessions(limit=9999)
        stats = {"total": len(sessions), "indexed": 0, "skipped": 0, "errors": 0}

        # Resolve model once for consistency
        active_model = model or self.model or _FAST_MODEL

        for i, session in enumerate(sessions):
            session_id = session.get("id", "")
            if not session_id:
                continue

            # Skip if already indexed (unless force)
            if not force:
                existing = self.get(session_id)
                if existing and existing.get("msg_count", 0) >= session.get("message_count", 0):
                    stats["skipped"] += 1
                    if progress_callback:
                        progress_callback(i + 1, len(sessions), session_id, "skipped")
                    continue

            if progress_callback:
                progress_callback(i + 1, len(sessions), session_id, "indexing")

            try:
                messages = db.get_messages(session_id, active_only=True)
                self.index_session(session, messages, force=force, model=active_model)
                stats["indexed"] += 1
            except Exception:
                stats["errors"] += 1

            if progress_callback:
                progress_callback(
                    i + 1, len(sessions), session_id,
                    "error" if stats.get("last_error") else "done",
                )

        return stats

    def index_new(self, builder: "ContextBuilder",  # noqa: F821
                  model: str | None = None) -> dict:
        """Index only sessions not yet in the index."""
        return self.index_all(builder, force=False, model=model)

    def clear(self) -> None:
        """Remove all entries from the index."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM session_index")
            conn.commit()
        finally:
            conn.close()

    def remove(self, session_id: str) -> None:
        """Remove a single session from the index."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM session_index WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()
