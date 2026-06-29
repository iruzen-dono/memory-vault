"""Memory Vault TUI — interactive session and pack browser.

Shows LLM-enriched session titles and summaries instead of raw
auto-generated names. Auto-indexes on first launch if the index
is empty.

Usage:
    memory-vault browse            # browse sessions
    memory-vault browse --packs    # browse packs in current directory

Uses Textual (https://textual.textualize.io) for a terminal UI.
"""

from __future__ import annotations

import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)

from memory_vault.core.builder import ContextBuilder
from memory_vault.core.llm import list_providers
from memory_vault.core.pack import HERMES_MEMORY_EXTENSION, ContextPack
from memory_vault.core.session_index import SessionIndex

# ── Provider status widget ───────────────────────────────────────────


class ProviderStatus(Static):
    """Shows the active LLM provider and its availability."""

    def on_mount(self) -> None:
        self.refresh_status()

    def refresh_status(self) -> None:
        try:
            statuses = list_providers()
            if not statuses:
                self.update("[dim]LLM: none[/]")
                return
            # Show first available, or first overall
            for name, avail in statuses.items():
                icon = "✅" if avail else "❌"
                self.update(f"[dim]LLM: {icon} {name}[/]")
                return
        except Exception:
            self.update("[dim]LLM: ?[/]")

# ── Session list widget ─────────────────────────────────────────────


class SessionItem(ListItem):
    """A single session entry in the list.

    Displays the LLM-enriched title (if indexed) with a summary preview.
    """

    def __init__(self, session: dict, indexed: dict | None = None) -> None:
        self.session = session
        self.indexed = indexed or {}
        sid = session["id"][:16]

        # Use enriched title if available, fall back to session title
        title = (indexed or {}).get("title") or session.get("title") or "(untitled)"
        summary = (indexed or {}).get("summary", "")

        model = session.get("model", "?")
        msgs = session.get("message_count", 0)

        # Build display
        if summary:
            # Truncate summary to fit one line
            summary_short = summary[:80] + ("…" if len(summary) > 80 else "")
            label = (
                f"[bold]{title}[/]  [dim]{sid}[/]\n"
                f"  {summary_short}  [dim]· {model} · {msgs} msgs[/]"
            )
        else:
            label = f"[bold]{title}[/]  [dim]{sid} · {model} · {msgs} msgs[/]"

        super().__init__(Label(label))


class PackItem(ListItem):
    """A single pack entry in the list."""

    def __init__(self, path: Path, summary: dict) -> None:
        self.pack_path = path
        self.summary = summary
        label = (
            f"[bold]{summary['title']}[/]  [dim]{path.name}[/]\n"
            f"  {summary['message_count']} msgs · {summary['artifact_count']} arts · "
            f"{summary['has_narrative'] and '📖' or ''} {summary['has_handoff'] and '🎯' or ''}"
        )
        super().__init__(Label(label))


# ── Detail panel ────────────────────────────────────────────────────


class DetailPanel(Vertical):
    """Right-side detail panel showing the selected item's info."""

    def show_session(self, session: dict, indexed: dict | None = None) -> None:
        indexed = indexed or {}
        title = indexed.get("title") or session.get("title") or "(untitled)"
        summary = indexed.get("summary", "")
        model_used = indexed.get("model_used", "")

        lines = [f"[bold]{title}[/]\n"]

        if summary:
            lines.append(f"[italic]{summary}[/]\n")

        lines.append(f"[dim]ID:[/] {session['id']}")
        lines.append(f"[dim]Source:[/] {session.get('source', '?')}")
        lines.append(f"[dim]Model:[/] {session.get('model', '?')}")
        lines.append(f"[dim]Messages:[/] {session.get('message_count', 0)}")
        lines.append(f"[dim]Created:[/] {session.get('started_at', '?')}")

        if model_used:
            lines.append(f"[dim]Indexed by:[/] {model_used}")

        lines.append("")
        lines.append("[italic]Press 'e' to export this session[/]")
        lines.append("[italic]Press 'i' to re-index this session[/]")

        self.remove_children()
        self.mount(Static("\n".join(lines)))

    def show_pack(self, pack_path: Path) -> None:
        try:
            pack = ContextPack.read(pack_path)
            s = pack.summary()
            self.remove_children()
            self.mount(
                Static(
                    f"[bold]{s['title']}[/]\n\n"
                    f"[dim]Type:[/] {s['pack_type']} v{s['format_version']}\n"
                    f"[dim]Author:[/] {s['author'] or '—'}\n"
                    f"[dim]Created:[/] {s['created_at']}\n"
                    f"[dim]Messages:[/] {s['message_count']}\n"
                    f"[dim]Artifacts:[/] {s['artifact_count']} files\n"
                    f"[dim]Decisions:[/] {s['decisions']}\n"
                    f"[dim]Narrative:[/] {'✅' if s['has_narrative'] else '❌'}\n"
                    f"[dim]Handoff:[/] {'✅' if s['has_handoff'] else '❌'}\n\n"
                    "[italic]Press 'i' to import this pack[/]"
                )
            )
        except Exception as e:
            self.remove_children()
            self.mount(Static(f"[red]Error: {e}[/]"))

    def show_message(self, text: str) -> None:
        self.remove_children()
        self.mount(Static(text))


# ── Session browser screen ─────────────────────────────────────────


class SessionBrowser(Screen):
    """Main screen: session list on the left, detail on the right.

    Loads the session index at startup and auto-indexes if empty.
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("e", "export", "Export"),
        ("i", "reindex", "Re-index"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._builder = ContextBuilder()
        self._index = SessionIndex()
        self._sessions: list[dict] = []
        self._indexed_map: dict[str, dict] = {}
        self._search_timer: float = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            Vertical(
                Label("[bold]Sessions[/]", id="list-title"),
                Input(placeholder="Search sessions...", id="search-input"),
                ProviderStatus(id="provider-status"),
                ListView(id="session-list"),
                id="left-panel",
            ),
            Vertical(
                Label("[bold]Details[/]", id="detail-title"),
                DetailPanel(id="detail-panel"),
                id="right-panel",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_sessions()

    @work
    async def load_sessions(self, query: str = "") -> None:
        """Load sessions and index, then populate the list."""
        list_view = self.query_one("#session-list", ListView)
        list_view.clear()
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.show_message("[dim]Loading sessions...[/]")

        try:
            # Load sessions from Hermes DB
            if query:
                self._sessions = self._builder.search_sessions(query, limit=50)
            else:
                self._sessions = self._builder.list_sessions(limit=50)

            if not self._sessions:
                detail.show_message("[yellow]No sessions found[/]")
                return

            # Load index and auto-index if empty
            self._load_index()

            # Populate list with enriched titles
            for s in self._sessions:
                sid = s["id"]
                indexed = self._indexed_map.get(sid)
                list_view.append(SessionItem(s, indexed))

            # Show first session
            list_view.index = 0
            self._show_current_session()
        except Exception as e:
            detail.show_message(f"[red]Error loading sessions: {e}[/]")

    def _load_index(self) -> None:
        """Load the session index map. Auto-indexes if empty."""
        if self._index.count() == 0:
            self._auto_index()
        else:
            for entry in self._index.list_indexed(limit=999):
                self._indexed_map[entry["session_id"]] = entry

    def _auto_index(self) -> None:
        """Index all sessions in the background (template fallback if no LLM)."""
        try:
            if not self._builder.db_path.exists():
                return
            from memory_vault.core.builder import HermesSessionDB
            db = HermesSessionDB(self._builder.db_path)
            sessions = db.list_sessions(limit=9999)

            for session in sessions:
                sid = session.get("id", "")
                if not sid:
                    continue
                try:
                    messages = db.get_messages(sid, active_only=True)
                    entry = self._index.index_session(session, messages)
                    if entry:
                        self._indexed_map[sid] = entry
                except Exception:
                    continue
        except Exception:
            pass

    def _show_current_session(self) -> None:
        """Show the currently highlighted session in the detail panel."""
        list_view = self.query_one("#session-list", ListView)
        if list_view.highlighted_child is None:
            return
        session = list_view.highlighted_child.session  # type: ignore
        sid = session["id"]
        indexed = self._indexed_map.get(sid)
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.show_session(session, indexed)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            # 200ms debounce
            self._search_timer = time.monotonic()
            self.set_timer(0.25, self._debounced_search)

    def _debounced_search(self) -> None:
        elapsed = time.monotonic() - self._search_timer
        if elapsed < 0.2:
            return  # Another keystroke arrived, skip this stale shot
        input_widget = self.query_one("#search-input", Input)
        query = input_widget.value.strip()
        self.load_sessions(query)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        if not hasattr(event.item, "session"):
            return
        session = event.item.session  # type: ignore
        sid = session["id"]
        indexed = self._indexed_map.get(sid)
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.show_session(session, indexed)

    def action_export(self) -> None:
        list_view = self.query_one("#session-list", ListView)
        if list_view.highlighted_child is None:
            return
        session = list_view.highlighted_child.session  # type: ignore
        session_id = session["id"]
        title = session.get("title") or "session"
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        out = Path.cwd() / f"{safe.lower().replace(' ', '-')}.hermes-memory"

        detail = self.query_one("#detail-panel", DetailPanel)
        detail.show_message(f"[dim]Exporting {session_id[:16]}...[/]")

        try:
            pack = self._builder.build_from_session(session_id=session_id)
            out_path = pack.write(out)
            detail.show_message(f"[green]✅ Exported → {out_path.name}[/]")
        except Exception as e:
            detail.show_message(f"[red]❌ Export failed: {e}[/]")

    def action_reindex(self) -> None:
        """Re-index the currently highlighted session."""
        list_view = self.query_one("#session-list", ListView)
        if list_view.highlighted_child is None:
            return
        session = list_view.highlighted_child.session  # type: ignore
        session_id = session["id"]
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.show_message(f"[dim]Re-indexing {session_id[:16]}...[/]")

        try:
            from memory_vault.core.builder import HermesSessionDB
            db = HermesSessionDB(self._builder.db_path)
            messages = db.get_messages(session_id, active_only=True)
            entry = self._index.index_session(session, messages, force=True)

            if entry:
                self._indexed_map[session_id] = entry
                # Refresh the list item
                idx = list_view.index
                if idx is not None:
                    list_view.pop(index=idx)
                    list_view.insert(idx, SessionItem(session, entry))
                detail.show_session(session, entry)
                detail.show_message(
                    f"[green]✅ Re-indexed →[/] [bold]{entry.get('title', '')}[/]"
                )
            else:
                detail.show_message("[yellow]⚠️  Re-index skipped (no data)[/]")
        except Exception as e:
            detail.show_message(f"[red]❌ Re-index failed: {e}[/]")

    def action_refresh(self) -> None:
        """Refresh the session list."""
        self.load_sessions()


# ── Pack browser screen ────────────────────────────────────────────


class PackBrowser(Screen):
    """Screen to browse .hermes-memory packs in a directory."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("i", "import_pack", "Import"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, directory: str | Path = ".") -> None:
        super().__init__()
        self.directory = Path(directory)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            Vertical(
                Label(f"[bold]Packs in {self.directory}[/]", id="pack-list-title"),
                ListView(id="pack-list"),
                id="left-panel",
            ),
            Vertical(
                Label("[bold]Details[/]", id="pack-detail-title"),
                DetailPanel(id="pack-detail-panel"),
                id="right-panel",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_packs()

    def load_packs(self) -> None:
        list_view = self.query_one("#pack-list", ListView)
        list_view.clear()
        detail = self.query_one("#pack-detail-panel", DetailPanel)

        pack_paths = sorted(self.directory.glob(f"*{HERMES_MEMORY_EXTENSION}"))
        if not pack_paths:
            detail.show_message(f"[yellow]No .hermes-memory packs in {self.directory}[/]")
            return

        for p in pack_paths:
            try:
                pack = ContextPack.read(p)
                list_view.append(PackItem(p, pack.summary()))
            except Exception:
                list_view.append(
                    ListItem(Label(f"[red]⚠ {p.name} (invalid)[/]"))
                )

        if pack_paths:
            list_view.index = 0
            detail.show_pack(pack_paths[0])

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        if not hasattr(event.item, "pack_path"):
            return
        detail = self.query_one("#pack-detail-panel", DetailPanel)
        detail.show_pack(event.item.pack_path)  # type: ignore

    def action_import_pack(self) -> None:
        list_view = self.query_one("#pack-list", ListView)
        if list_view.highlighted_child is None:
            return
        if not hasattr(list_view.highlighted_child, "pack_path"):
            return
        pack_path = list_view.highlighted_child.pack_path  # type: ignore
        detail = self.query_one("#pack-detail-panel", DetailPanel)

        try:
            pack = ContextPack.read(pack_path)
            out_dir = Path.cwd() / "imported"
            out_dir.mkdir(exist_ok=True)

            # Write handoff and narrative
            if pack.handoff_md:
                (out_dir / "handoff.md").write_text(pack.handoff_md, encoding="utf-8")
            if pack.narrative_md:
                (out_dir / "narrative.md").write_text(pack.narrative_md, encoding="utf-8")

            detail.show_message(
                f"[green]✅ Imported → {out_dir}/\n"
                f"   handoff.md {'✅' if pack.handoff_md else '❌'} · "
                f"narrative.md {'✅' if pack.narrative_md else '❌'}[/]"
            )
        except Exception as e:
            detail.show_message(f"[red]❌ Import failed: {e}[/]")

    def action_refresh(self) -> None:
        self.load_packs()


# ── Main app ────────────────────────────────────────────────────────


class MemoryVaultTUI(App):
    """Memory Vault TUI — browse sessions and packs interactively."""

    TITLE = "Memory Vault"
    SUB_TITLE = "Portable Context Protocol"
    CSS = """
    Screen {
        layout: vertical;
    }
    Horizontal {
        height: 1fr;
    }
    #left-panel, #right-panel {
        padding: 1;
        border: solid $border;
    }
    #left-panel {
        width: 2fr;
    }
    #right-panel {
        width: 3fr;
    }
    #search-input {
        margin: 0 0 1 0;
    }
    #provider-status {
        margin: 0 0 1 0;
        text-style: dim;
    }
    ListView {
        height: 1fr;
    }
    DetailPanel {
        height: 1fr;
    }
    #list-title, #detail-title, #pack-list-title, #pack-detail-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, mode: str = "sessions", directory: str | Path = ".") -> None:
        super().__init__()
        self.mode = mode
        self.directory = Path(directory)

    def on_mount(self) -> None:
        if self.mode == "packs":
            self.push_screen(PackBrowser(self.directory))
        else:
            self.push_screen(SessionBrowser())


def browse(mode: str = "sessions", directory: str | Path = ".") -> None:
    """Launch the Memory Vault TUI."""
    app = MemoryVaultTUI(mode=mode, directory=directory)
    app.run()
