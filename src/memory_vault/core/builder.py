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

from .manifest import (
    PACK_TYPE_DAILY,
    PACK_TYPE_PROJECT,
    ArtifactIndex,
    Manifest,
    NarrativeIndex,
    ToolUsage,
)
from .narrator import SessionNarrator
from .pack import ContextPack

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

        # Build tool traces summary
        tool_traces = {
            "total_tool_calls": sum(tool_counts.values()),
            "unique_tools": list(tool_counts.keys()),
            "by_tool": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
            "artifact_tools_used": {
                t: tool_counts[t] for t in _ARTIFACT_TOOLS if t in tool_counts
            },
        }

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

        # Read git commit SHA (best-effort)
        source_commit = self._read_git_commit()

        # Build manifest
        manifest = Manifest(
            title=title or session_title,
            description=description,
            tags=tags or [],
            author=author,
            source_session_id=session_id,
            source_platform=session.get("source", ""),
            source_model=session.get("model", ""),
            source_commit=source_commit,
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

    # -- Project pack builder ------------------------------------------

    def build_project_pack(
        self,
        project_dir: str,
        title: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        author: str = "",
        include_env: bool = False,
        tree_depth: int = 2,
    ) -> ContextPack:
        """Capture a project snapshot as a project-pack.

        Records the file tree, key files (README, config), and env template.
        Useful for onboarding context or project backup.
        """
        proj = Path(project_dir).resolve()
        proj_name = title or proj.name

        # Generate directory tree
        tree = self._generate_tree(proj, depth=tree_depth)

        # Collect key files as artifacts
        artifacts: dict[str, Path] = {}
        key_patterns = ["README.md", "README.txt", "package.json", "Cargo.toml",
                        "pyproject.toml", "composer.json", "Makefile",
                        ".env.example", ".env.sample", "docker-compose.yml",
                        "Dockerfile", "tsconfig.json", ".editorconfig"]

        if include_env:
            key_patterns.extend([".env.example", ".env.sample"])

        for pattern in key_patterns:
            for f in proj.rglob(pattern):
                if f.is_file() and ".git" not in f.parts:
                    rel = f.relative_to(proj)
                    artifacts[str(rel)] = f

        # Also include config directories
        for cfg_dir in [".github", ".vscode", ".cursor"]:
            cfg_path = proj / cfg_dir
            if cfg_path.is_dir():
                for f in sorted(cfg_path.rglob("*")):
                    if f.is_file() and ".git" not in f.parts:
                        rel = f.relative_to(proj)
                        artifacts[str(rel)] = f

        # Read README as narrative (if exists)
        readme = proj / "README.md"
        narrative = ""
        if readme.is_file():
            narrative = f"# {proj_name}\n\n{readme.read_text(encoding='utf-8', errors='replace')}"
        else:
            narrative = f"# {proj_name}\n\n*No README found.*\n"

        # Build manifest
        manifest = Manifest(
            title=proj_name,
            description=description or f"Project snapshot: {proj_name}",
            tags=tags or [],
            author=author,
            pack_type=PACK_TYPE_PROJECT,
            pack_subtype="project-structure",
            project_name=proj_name,
            project_path=str(proj),
            artifacts=ArtifactIndex(
                count=len(artifacts),
                files=sorted(artifacts.keys()),
            ),
            source_commit=self._read_git_commit(),
        )

        return ContextPack(
            manifest=manifest,
            narrative_md=narrative,
            artifacts=artifacts,
            tool_traces={"total_tool_calls": 0, "unique_tools": [], "by_tool": {}},
            handoff_md=f"# Project Context: {proj_name}\n\n"
                       f"Project directory: `{proj}`\n\n"
                       f"Key files captured. Use `memory-vault import --restore <dir>` "
                       f"to restore artifacts locally.",
            references_md="",
            directory_tree=tree,
        )

    # -- Daily pack builder --------------------------------------------

    def build_daily_pack(
        self,
        date_str: str | None = None,
        limit: int = 50,
        description: str = "",
        author: str = "",
    ) -> ContextPack:
        """Bundle today's sessions into a daily context pack.

        Scans Hermes sessions for today and merges their context
        into a single daily-pack.
        """
        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                target_date = datetime.now().date()
        else:
            target_date = datetime.now().date()

        # Get all sessions
        all_sessions = self.list_sessions(limit=limit)

        # Filter sessions from target date
        day_sessions = []
        for s in all_sessions:
            started = s.get("started_at")
            if started is None:
                continue
            try:
                ts = int(started) if isinstance(started, (int, float)) else 0
                session_date = datetime.fromtimestamp(ts).date()
                if session_date == target_date:
                    day_sessions.append(s)
            except (ValueError, OSError):
                continue

        title = f"Daily Digest — {target_date.isoformat()}"

        # Merge session info
        total_msgs = sum(s.get("message_count", 0) for s in day_sessions)
        total_tools = sum(s.get("tool_call_count", 0) for s in day_sessions)
        sources = list(set(s.get("source", "?") for s in day_sessions))
        models = list(set(s.get("model", "?") for s in day_sessions))

        # Build narrative
        lines = [f"# {title}", ""]
        lines.append(f"**Sessions:** {len(day_sessions)}")
        lines.append(f"**Messages:** {total_msgs}")
        lines.append(f"**Tool calls:** {total_tools}")
        lines.append(f"**Sources:** {', '.join(sources)}")
        if models:
            lines.append(f"**Models:** {', '.join(models)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for i, s in enumerate(day_sessions, 1):
            sid = s["id"][:16] if s.get("id") else "?"
            stitle = s.get("title", "Untitled")
            lines.append(f"## {i}. {stitle}")
            lines.append(f"   📍 {s.get('source', '?')} · {s.get('model', '?')}")
            lines.append(f"   Msgs: {s.get('message_count', 0)} · Tools: {s.get('tool_call_count', 0)}")
            lines.append(f"   🆔 {sid}")
            lines.append("")

        narrative = "\n".join(lines)

        handoff = (
            f"# Handoff: {title}\n\n"
            f"This daily pack captures {len(day_sessions)} session(s) "
            f"from {target_date.isoformat()}.\n\n"
            f"Total messages: {total_msgs} · Tool calls: {total_tools}\n\n"
            f"Review narrative.md for the full digest."
        )

        manifest = Manifest(
            title=title,
            description=description or f"Daily session digest for {target_date.isoformat()}",
            tags=["daily", target_date.isoformat()],
            author=author,
            pack_type=PACK_TYPE_DAILY,
            pack_subtype="daily-timeline",
            message_count=total_msgs,
            source_commit=self._read_git_commit(),
        )

        return ContextPack(
            manifest=manifest,
            narrative_md=narrative,
            messages=[],
            decisions=[],
            artifacts={},
            tool_traces={"total_tool_calls": total_tools, "unique_tools": [], "by_tool": {}},
            handoff_md=handoff,
            references_md="",
        )

    # -- Directory tree generator --------------------------------------

    @staticmethod
    def _generate_tree(directory: Path, depth: int = 2) -> str:
        """Generate a directory tree string up to a given depth."""
        lines = [f"{directory.name}/"]
        prefix = "    "

        def _walk(dirpath: Path, level: int = 0):
            if level > depth:
                return
            try:
                entries = sorted(dirpath.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except PermissionError:
                return

            for i, entry in enumerate(entries):
                if entry.name.startswith(".") and entry.name not in (".github", ".vscode", ".cursor"):
                    continue
                if entry.name == "__pycache__":
                    continue
                if ".git" in entry.parts:
                    continue

                is_last = i == len(entries) - 1
                branch = "└── " if is_last else "├── "
                lines.append(prefix * level + branch + entry.name)

                if entry.is_dir():
                    _walk(entry, level + 1)

        _walk(directory)
        return "\n".join(lines)

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

    # -- Git commit SHA detection --------------------------------------

    @staticmethod
    def _read_git_commit() -> str:
        """Best-effort read of the current git commit SHA."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                return result.stdout.strip()[:40]
        except Exception:
            pass
        return ""
