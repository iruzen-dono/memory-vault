"""ContextBuilder — builds ContextPacks from Hermes session data.

Reads a Hermes session (sessions.db), extracts the conversation,
detects artifacts, counts tool usage, and assembles everything
into a portable .hermes-memory context pack.

This is the engine that turns a session into shareable context.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .manifest import Manifest, ToolUsage, ArtifactIndex, NarrativeIndex
from .pack import ContextPack
from .narrator import SessionNarrator


# Tools that create or modify files — we track these as artifacts
_ARTIFACT_TOOLS = {"write_file", "patch"}


class HermesSessionDB:
    """Read-only interface to the Hermes sessions database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def list_sessions(
        self,
        limit: int = 20,
        source: str | None = None,
    ) -> list[dict]:
        """List recent sessions with metadata."""
        conn = self._connect()
        try:
            sql = (
                "SELECT id, title, source, model, started_at, ended_at, "
                "message_count, tool_call_count, cwd, git_repo_root "
                "FROM sessions "
                "WHERE title IS NOT NULL "
            )
            params = []
            if source:
                sql += "AND source = ? "
                params.append(source)
            sql += "ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search_sessions(self, query: str, limit: int = 10) -> list[dict]:
        """Search sessions by title using FTS5."""
        conn = self._connect()
        try:
            # Try FTS5 on messages first, then fall back to LIKE on title
            try:
                sql = (
                    "SELECT DISTINCT s.id, s.title, s.source, s.model, "
                    "s.started_at, s.ended_at, s.message_count "
                    "FROM sessions s "
                    "JOIN messages m ON m.session_id = s.id "
                    "JOIN messages_fts fts ON fts.rowid = m.id "
                    "WHERE messages_fts MATCH ? "
                    "ORDER BY s.started_at DESC LIMIT ?"
                )
                rows = conn.execute(sql, (query, limit)).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass

            # Fallback: title LIKE
            sql = (
                "SELECT id, title, source, model, "
                "started_at, ended_at, message_count "
                "FROM sessions "
                "WHERE title LIKE ? "
                "ORDER BY started_at DESC LIMIT ?"
            )
            rows = conn.execute(sql, (f"%{query}%", limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> dict | None:
        """Get session metadata by ID."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_messages(
        self,
        session_id: str,
        active_only: bool = True,
    ) -> list[dict]:
        """Get all messages for a session, ordered chronologically."""
        conn = self._connect()
        try:
            sql = (
                "SELECT id, session_id, role, content, tool_call_id, "
                "tool_calls, tool_name, timestamp, reasoning, "
                "reasoning_content "
                "FROM messages "
                "WHERE session_id = ? "
            )
            params = [session_id]
            if active_only:
                sql += "AND active = 1 "
            sql += "ORDER BY id ASC"

            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


class ContextBuilder:
    """Builds a ContextPack from a Hermes session.

    Typical usage::

        builder = ContextBuilder(hermes_home=...)
        pack = builder.build_from_session(
            session_id="abc123",
            title="Built Hyperliquid Trading Bot",
            tags=["trading", "hyperliquid"],
            author="iruzen",
        )
        pack.write("trading-bot.hermes-memory")
    """

    def __init__(self, hermes_home: str | Path | None = None):
        self.hermes_home = Path(hermes_home) if hermes_home else self._detect_hermes_home()
        self.db_path = self.hermes_home / "state.db"

    @staticmethod
    def _detect_hermes_home() -> Path:
        """Detect Hermes home directory (env var, then platform defaults)."""
        env = os.environ.get("HERMES_HOME")
        if env:
            return Path(env)
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "")
            if local_appdata:
                return Path(local_appdata) / "hermes"
            return Path.home() / "AppData" / "Local" / "hermes"
        return Path.home() / ".hermes"

    # -- Public API ---------------------------------------------------

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent Hermes sessions."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Hermes sessions DB not found at {self.db_path}"
            )
        db = HermesSessionDB(self.db_path)
        return db.list_sessions(limit=limit)

    def search_sessions(self, query: str, limit: int = 10) -> list[dict]:
        """Search sessions by keyword."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Hermes sessions DB not found at {self.db_path}"
            )
        db = HermesSessionDB(self.db_path)
        return db.search_sessions(query, limit=limit)

    def build_from_session(
        self,
        session_id: str,
        title: str = "",
        description: str = "",
        tags: list[str] | None = None,
        author: str = "",
        include_artifacts: bool = True,
        project_root: str | Path | None = None,
        narrate: bool = False,
        deep: bool = False,
    ) -> ContextPack:
        """Build a ContextPack from a Hermes session.

        Args:
            session_id: The Hermes session ID.
            title: Override title (defaults to session title).
            description: Short description of the action.
            tags: Tags for the pack.
            author: Author name.
            include_artifacts: If True, copy referenced files into the pack.

        Returns:
            A ready-to-write ContextPack.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Hermes sessions DB not found at {self.db_path}. "
                "Cannot build from session without a Hermes installation."
            )

        db = HermesSessionDB(self.db_path)
        session = db.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session_title = session.get("title") or ""

        # Read messages
        messages = db.get_messages(session_id, active_only=True)

        # Extract tool calls and detect artifacts
        tool_counts: dict[str, int] = {}
        artifact_paths: dict[str, Path] = {}
        artifact_sources: set[str] = set()

        for msg in messages:
            # Count tool usage from tool_name
            tname = msg.get("tool_name")
            if tname:
                tool_counts[tname] = tool_counts.get(tname, 0) + 1

            # Also count from tool_calls JSON (assistant messages)
            tc_raw = msg.get("tool_calls")
            if tc_raw:
                try:
                    tool_calls = json.loads(tc_raw) if isinstance(tc_raw, str) else tc_raw
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            name = tc.get("name", tc.get("function", {}).get("name", ""))
                            if name:
                                tool_counts[name] = tool_counts.get(name, 0) + 1

                            # Detect artifact tools
                            if name in _ARTIFACT_TOOLS:
                                args_raw = tc.get("arguments", tc.get("function", {}).get("arguments", "{}"))
                                if isinstance(args_raw, str):
                                    try:
                                        args = json.loads(args_raw)
                                    except json.JSONDecodeError:
                                        continue
                                else:
                                    args = args_raw
                                file_path = args.get("path", "")
                                if file_path and file_path not in artifact_sources:
                                    artifact_sources.add(file_path)
                                    fp = Path(file_path)
                                    if fp.exists():
                                        resolved = fp.resolve()
                                        rel = None
                                        # 1) Try explicit project_root
                                        if project_root:
                                            try:
                                                rel = str(resolved.relative_to(Path(project_root).resolve()))
                                            except ValueError:
                                                pass
                                        # 2) Try session cwd
                                        if not rel:
                                            session_cwd = session.get("cwd", "")
                                            if session_cwd:
                                                try:
                                                    rel = str(resolved.relative_to(Path(session_cwd).resolve()))
                                                except ValueError:
                                                    pass
                                        # 3) Try git root
                                        if not rel:
                                            git_root = session.get("git_repo_root", "")
                                            if git_root:
                                                try:
                                                    rel = str(resolved.relative_to(Path(git_root).resolve()))
                                                except ValueError:
                                                    pass
                                        # 4) Fallback: just the filename
                                        if not rel:
                                            rel = fp.name
                                        artifact_paths[rel] = resolved
                except (json.JSONDecodeError, TypeError):
                    pass

        # Calculate duration
        duration_min = 0
        started = session.get("started_at")
        ended = session.get("ended_at")
        if started and ended:
            duration_min = int((ended - started) / 60) if ended > started else 0

        # Build narrative (template or LLM-compressed)
        if narrate:
            narrator = SessionNarrator()
            nar_result = narrator.summarize(session, messages, tool_traces, deep=deep)
            narrative_md = nar_result.summary_md
            handoff_md = nar_result.handoff_md
            decisions = nar_result.decisions
        else:
            narrative_md = self._build_narrative(session, messages)
            handoff_md = self._build_handoff(session, messages)
            decisions = []

        # Build references
        references_md = self._build_references(messages)

        # Build tool traces summary
        tool_traces = {
            "total_tool_calls": sum(tool_counts.values()),
            "unique_tools": list(tool_counts.keys()),
            "by_tool": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
            "artifact_tools_used": {
                t: tool_counts[t] for t in _ARTIFACT_TOOLS if t in tool_counts
            },
        }

        # Build manifest
        manifest = Manifest(
            title=title or session_title,
            description=description,
            tags=tags or [],
            author=author,
            source_session_id=session_id,
            source_platform=session.get("source", ""),
            source_model=session.get("model", ""),
            duration_minutes=duration_min,
            message_count=len(messages),
            narrative=NarrativeIndex(chapters=[]),  # reserved for AI-summarized chapters
            artifacts=ArtifactIndex(
                count=len(artifact_paths),
                files=sorted(artifact_paths.keys()),
            ),
            tool_usage=ToolUsage(
                total_calls=sum(tool_counts.values()),
                by_tool=dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
            ),
        )

        return ContextPack(
            manifest=manifest,
            narrative_md=narrative_md,
            messages=messages,
            decisions=decisions,  # AI-extracted decisions (or empty)
            artifacts=artifact_paths if include_artifacts else {},
            tool_traces=tool_traces,
            handoff_md=handoff_md,
            references_md=references_md,
        )

    # -- Narrative generation -----------------------------------------

    def _build_narrative(self, session: dict, messages: list[dict]) -> str:
        """Build a chronological narrative from session messages."""
        lines = [
            f"# {session.get('title') or 'Session Narrative'}",
            "",
        ]

        # Session header
        started = session.get("started_at")
        if started:
            try:
                dt = datetime.fromtimestamp(started)
                lines.append(f"**Started:** {dt.strftime('%Y-%m-%d %H:%M UTC')}")
            except (ValueError, OSError):
                lines.append(f"**Started:** {started}")
        if session.get("model"):
            lines.append(f"**Model:** {session['model']}")
        if session.get("source"):
            lines.append(f"**Platform:** {session['source']}")
        lines.append(f"**Messages:** {session.get('message_count', len(messages))}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Chronological transcript
        user_msg_count = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                user_msg_count += 1
                lines.append(f"## Message {user_msg_count}")
                lines.append("")
                lines.append(content if content else "*[empty]*")
                lines.append("")

            elif role == "assistant":
                # Show reasoning if present
                reasoning = msg.get("reasoning") or msg.get("reasoning_content", "")
                if reasoning:
                    lines.append("**Reasoning:**")
                    lines.append("")
                    lines.append(reasoning)
                    lines.append("")

                # Show tool calls if present
                tc_raw = msg.get("tool_calls")
                if tc_raw:
                    try:
                        tool_calls = json.loads(tc_raw) if isinstance(tc_raw, str) else tc_raw
                        if isinstance(tool_calls, list):
                            for tc in tool_calls:
                                name = tc.get("name", tc.get("function", {}).get("name", ""))
                                lines.append(f"> **Tool:** `{name}`")
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Show assistant response
                if content:
                    lines.append(content)
                    lines.append("")

            elif role == "tool":
                # Only include notable tool results (short ones)
                tname = msg.get("tool_name", "")
                short_content = ""
                if content and len(content) > 200:
                    short_content = content[:200] + "..."
                else:
                    short_content = content
                if short_content:
                    lines.append(f"*Tool `{tname}` returned:*")
                    lines.append(f"> {short_content}")
                    lines.append("")

        return "\n".join(lines)

    # -- Handoff generation -------------------------------------------

    def _build_handoff(self, session: dict, messages: list[dict]) -> str:
        """Build a handoff brief for the receiving agent."""
        title = session.get("title") or "Untitled Session"
        system_prompt = session.get("system_prompt", "")
        model = session.get("model", "unknown")

        # Find the first user message (the goal)
        first_goal = ""
        for msg in messages:
            if msg.get("role") == "user":
                first_goal = msg.get("content", "")
                break

        # Find the last assistant message (the conclusion)
        last_result = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    last_result = content[:500]
                    break

        lines = [
            f"# Context Handoff: {title}",
            "",
            "This pack captures a complete action from an AI agent session.",
            "Load this context to understand what was done, how, and why.",
            "",
            "---",
            "",
            "## Original Goal",
            "",
            first_goal or "*Unknown*",
            "",
            "## Key Results",
            "",
            last_result or "*Session in progress*",
            "",
            "## Session Metadata",
            "",
            f"- **Model:** {model}",
            f"- **Platform:** {session.get('source', 'unknown')}",
        ]

        # System prompt overview
        if system_prompt:
            # Extract first meaningful lines
            sp_lines = system_prompt.strip().split("\n")[:10]
            lines.append("")
            lines.append("## System Context")
            lines.append("")
            for sl in sp_lines:
                lines.append(f"> {sl}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("To continue this work, load this pack and review the")
        lines.append("narrative and tool traces to understand the full context.")

        return "\n".join(lines)

    # -- References extraction ----------------------------------------

    def _build_references(self, messages: list[dict]) -> str:
        """Extract links, API docs, and resources mentioned in the session."""
        urls: list[str] = []
        for msg in messages:
            content = msg.get("content", "") or ""
            # Simple URL extraction
            import re
            found = re.findall(r'https?://[^\s)]+', content)
            urls.extend(found)

        # Deduplicate
        urls = list(dict.fromkeys(urls))

        if not urls:
            return "# References\n\n*No URLs found in this session.*\n"

        lines = ["# References", ""]
        for i, url in enumerate(urls, 1):
            lines.append(f"{i}.  {url}")
        lines.append("")
        return "\n".join(lines)
