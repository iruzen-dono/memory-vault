"""Memory Vault CLI — package and share agent context.

Usage:
    memory-vault list-sessions              # browse sessions
    memory-vault export <session-id>        # pack a session into .hermes-memory
    memory-vault info <pack>                # inspect a pack
    memory-vault import <pack>              # extract handoff & context
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from memory_vault import __version__
from memory_vault.core.builder import ContextBuilder
from memory_vault.core.pack import ContextPack, HERMES_MEMORY_EXTENSION

app = typer.Typer(
    name="memory-vault",
    help="Portable Context Protocol — package and share AI agent actions",
    no_args_is_help=True,
)


@app.callback()
def version_callback(version: bool = typer.Option(False, "--version", "-V", help="Show version")):
    if version:
        typer.echo(f"memory-vault v{__version__}")
        raise typer.Exit()


# ── list-sessions: browse Hermes sessions ─────────────────────────


@app.command(name="list-sessions")
def list_sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search sessions by keyword"),
    hermes_home: Optional[str] = typer.Option(None, "--hermes-home", help="Custom Hermes home path"),
):
    """Browse recent Hermes sessions available for packaging."""
    builder = ContextBuilder(hermes_home=hermes_home)

    try:
        if search:
            sessions = builder.search_sessions(search, limit=limit)
        else:
            sessions = builder.list_sessions(limit=limit)
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(1)

    if not sessions:
        typer.echo("No sessions found.")
        raise typer.Exit()

    heading = f"Recent sessions matching \"{search}\"" if search else "Recent sessions"
    typer.echo(f"📋 {heading}\n")
    for s in sessions:
        sid = s["id"]
        title = s.get("title") or "(untitled)"
        source = s.get("source", "?")
        model = s.get("model", "?")
        msgs = s.get("message_count", 0)
        # Shorten session ID for display
        typer.echo(f"  [{sid}] {title}")
        typer.echo(f"           {source} · {model} · {msgs} messages")
        typer.echo("")


# ── export: build a context pack from a session ───────────────────


@app.command()
def export(
    session_id: str = typer.Argument(..., help="Hermes session ID to package"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Pack title (default: session title)"),
    description: str = typer.Option("", "--description", "-d", help="Short description of the action"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    author: str = typer.Option("", "--author", "-a", help="Author name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output .hermes-memory file path"),
    no_artifacts: bool = typer.Option(False, "--no-artifacts", help="Exclude file artifacts from the pack"),
    hermes_home: Optional[str] = typer.Option(None, "--hermes-home", help="Custom Hermes home path"),
    project_root: Optional[str] = typer.Option(None, "--project-root", "-r", help="Base directory for artifact relative paths"),
):
    """Package a Hermes session into a portable .hermes-memory context pack.

    Captures the full context: conversation, tool usage, artifacts,
    decisions, and a handoff brief for another agent.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    builder = ContextBuilder(hermes_home=hermes_home)

    try:
        pack = builder.build_from_session(
            session_id=session_id,
            title=title or "",
            description=description,
            tags=tag_list,
            author=author,
            include_artifacts=not no_artifacts,
            project_root=project_root,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(1)

    # Default output path
    if output is None:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in (title or pack.manifest.title or "context"))
        output = Path.cwd() / f"{safe_title.lower().replace(' ', '-')}{HERMES_MEMORY_EXTENSION}"

    out_path = pack.write(output)
    s = pack.summary()
    typer.echo(f"✅ Context pack → {out_path}")
    typer.echo(f"   Title:    {s['title']}")
    typer.echo(f"   Messages: {s['message_count']}")
    typer.echo(f"   Artifacts: {s['artifact_count']} files")
    typer.echo(f"   Tools:    {pack.tool_traces.get('total_tool_calls', 0)} calls ({' '.join(pack.tool_traces.get('unique_tools', []))})")
    typer.echo(f"   Narrative: {'✅' if s['has_narrative'] else '❌'}")
    typer.echo(f"   Handoff:   {'✅' if s['has_handoff'] else '❌'}")


# ── info: inspect a pack ──────────────────────────────────────────


@app.command()
def info(
    path: Path = typer.Argument(..., help="Path to .hermes-memory file", exists=True),
):
    """Inspect a .hermes-memory context pack without importing."""
    pack = ContextPack.read(path)
    s = pack.summary()

    typer.echo(f"📦 {s['title']}")
    typer.echo(f"   Type:           {s['pack_type']}")
    typer.echo(f"   Description:    {s['description'] or '—'}")
    typer.echo(f"   Author:         {s['author'] or '—'}")
    typer.echo(f"   Created:        {s['created_at']}")
    typer.echo(f"   Format version: {s['format_version']}")
    typer.echo(f"   Tags:           {', '.join(s['tags']) or '—'}")
    typer.echo("")
    if s["source_session_id"]:
        typer.echo(f"   🔗 Source session: {s['source_session_id'][:16]}…")
    typer.echo(f"   💬 Messages:        {s['message_count']}")
    typer.echo(f"   📄 Artifacts:       {s['artifact_count']} files")
    if s["artifact_files"]:
        for af in s["artifact_files"][:5]:
            typer.echo(f"                       • {af}")
        if len(s["artifact_files"]) > 5:
            typer.echo(f"                       … and {len(s['artifact_files']) - 5} more")
    typer.echo(f"   🧠 Decisions:       {s['decisions']}")
    typer.echo(f"   🛠  Narrative:       {'✅' if s['has_narrative'] else '❌'}")
    typer.echo(f"   🎯 Handoff brief:   {'✅' if s['has_handoff'] else '❌'}")


# ── import: extract context from a pack ───────────────────────────


@app.command()
def import_pack(
    path: Path = typer.Argument(..., help="Path to .hermes-memory file", exists=True),
    extract_handoff: bool = typer.Option(True, "--handoff/--no-handoff", help="Show handoff brief"),
    list_artifacts: bool = typer.Option(True, "--list-artifacts/--no-artifacts", help="List available artifacts"),
):
    """Extract context from a .hermes-memory pack.

    Displays the handoff brief, available artifacts, and tool traces
    so the receiving agent can pick up where the session left off.
    """
    pack = ContextPack.read(path)
    s = pack.summary()

    typer.echo(f"📦 {s['title']}")
    typer.echo(f"   By: {s['author'] or 'unknown'} · {s['created_at']}")
    typer.echo("")

    if extract_handoff and pack.handoff_md:
        typer.echo("─" * 50)
        typer.echo("🎯 HANDOFF BRIEF")
        typer.echo("─" * 50)
        typer.echo("")
        typer.echo(pack.handoff_md)
        typer.echo("")

    if list_artifacts and pack.artifacts:
        typer.echo("─" * 50)
        typer.echo("📄 ARTIFACTS")
        typer.echo("─" * 50)
        for fpath in sorted(pack.artifacts.keys()):
            typer.echo(f"   • {fpath}")
        typer.echo("")

    if pack.tool_traces:
        typer.echo("─" * 50)
        typer.echo("🛠  TOOL USAGE")
        typer.echo("─" * 50)
        by_tool = pack.tool_traces.get("by_tool", {})
        for tool_name, count in by_tool.items():
            bar = "█" * min(count, 40)
            typer.echo(f"   {tool_name:25s} {bar} {count}")
        typer.echo("")

    typer.echo("─" * 50)
    typer.echo("ℹ️  To use this context in a new session:")
    typer.echo("   Pass the handoff brief as your initial context.")
    typer.echo("   The narrative.md and messages.json contain the full story.")


# ── list-packs: browse packs in a directory ───────────────────────


@app.command(name="list")
def list_packs(
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Directory to scan"),
):
    """List all .hermes-memory packs in a directory."""
    packs = list(path.glob(f"*{HERMES_MEMORY_EXTENSION}"))
    if not packs:
        typer.echo(f"No .hermes-memory packs found in {path}")
        raise typer.Exit()

    typer.echo(f"📁 Packs in {path}:\n")
    for p in sorted(packs):
        try:
            pack = ContextPack.read(p)
            s = pack.summary()
            typer.echo(f"  📦 {p.name}")
            typer.echo(f"      Title:   {s['title']}")
            typer.echo(f"      Type:    {s['pack_type']}")
            typer.echo(f"      Author:  {s['author'] or '—'}")
            typer.echo(f"      Size:    {p.stat().st_size / 1024:.1f} KB")
            typer.echo(f"      Msgs:    {s['message_count']} · Arts: {s['artifact_count']} · Tools: {pack.tool_traces.get('total_tool_calls', 0)}")
            typer.echo("")
        except Exception as e:
            typer.echo(f"  ⚠️  {p.name}: {e}")
            typer.echo("")


def main():
    app()
