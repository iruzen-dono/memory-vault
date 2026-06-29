"""Memory Vault CLI — package and share agent context.

Usage:
    memory-vault list-sessions              # browse sessions
    memory-vault export <session-id>        # pack a session into .hermes-memory
    memory-vault info <pack>                # inspect a pack
    memory-vault import <pack>              # extract handoff & context
    memory-vault list                       # browse packs in a directory
    memory-vault search <pattern>           # search across packs
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path
from typing import Optional

import typer

from memory_vault import __version__
from memory_vault.core.builder import ContextBuilder
from memory_vault.core.pack import HERMES_MEMORY_EXTENSION, ContextPack

app = typer.Typer(
    name="memory-vault",
    help="Portable Context Protocol — package and share AI agent actions",
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Manage memory-vault configuration",
    no_args_is_help=True,
)
app.add_typer(config_app)


@app.callback()
def version_callback(version: bool = typer.Option(False, "--version", "-V", help="Show version")):
    if version:
        typer.echo(f"memory-vault v{__version__}")
        raise typer.Exit()


# ── providers: show available LLM providers ────────────────────────


@app.command()
def providers():
    """List available LLM providers and their status."""
    from memory_vault.core.llm import list_providers

    statuses = list_providers()
    if not statuses:
        typer.echo("⚠️  No providers registered.")
        raise typer.Exit()

    typer.echo(f"  {'Provider':<16} {'Available':<12}")
    typer.echo(f"  {'─' * 16} {'─' * 12}")
    for name, available in statuses.items():
        icon = "✅" if available else "❌"
        typer.echo(f"  {name:<16} {icon:<12}")
    typer.echo("")
    typer.echo("Tip: set MEMORY_VAULT_LLM_PROVIDER=anthropic or run:")
    typer.echo("  memory-vault config set llm.provider anthropic")


# ── config: manage configuration ────────────────────────────────────


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. 'llm.provider')"),
    value: str = typer.Argument(..., help="Config value (e.g. 'anthropic')"),
):
    """Set a configuration value (persisted to ~/.config/memory-vault/config.yaml)."""
    from memory_vault.core.config import set_config_value

    set_config_value(key, value)
    typer.echo(f"✅ {key} = {value}")


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key (e.g. 'llm.provider')"),
):
    """Get a configuration value."""
    from memory_vault.core.config import get_config_value

    val = get_config_value(key)
    if val is not None:
        typer.echo(str(val))
    else:
        typer.echo("(not set)")
        raise typer.Exit(1)


@config_app.command("list")
def config_list():
    """Show all configuration values."""
    from memory_vault.core.config import read_config

    cfg = read_config()
    if not cfg:
        typer.echo("(empty — no config file yet)")
        raise typer.Exit()

    def _dump(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                typer.echo(f"  {prefix}{k}:")
                _dump(v, prefix + "  ")
            else:
                typer.echo(f"  {prefix}{k}: {v}")
    _dump(cfg)


# ── diff: compare two context packs ──────────────────────────────────


@app.command()
def diff(
    pack_a: str = typer.Argument(..., help="First .hermes-memory pack"),
    pack_b: str = typer.Argument(..., help="Second .hermes-memory pack"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full diff"),
):
    """Compare two context packs and show differences."""
    pa, pb = Path(pack_a), Path(pack_b)
    if not pa.exists():
        typer.echo(f"❌ Pack not found: {pa}", err=True)
        raise typer.Exit(1)
    if not pb.exists():
        typer.echo(f"❌ Pack not found: {pb}", err=True)
        raise typer.Exit(1)

    pack1 = ContextPack.read(pa)
    pack2 = ContextPack.read(pb)
    s1 = pack1.summary()
    s2 = pack2.summary()

    typer.echo(f"  {'Field':<24} {pa.name:<36} {pb.name:<36}")
    typer.echo(f"  {'─' * 24} {'─' * 36} {'─' * 36}")

    diff_fields = [
        ("Title", "title", str, str),
        ("Description", "description", str, str),
        ("Source session", "source_session_id", str, str),
        ("Messages", "message_count", int, int),
        ("Artifacts", "artifact_count", int, int),
        ("Decisions", "decisions", int, int),
        ("Format version", "format_version", str, str),
    ]

    changed = 0
    for label, key, *_ in diff_fields:
        v1 = s1.get(key, "")
        v2 = s2.get(key, "")
        same = v1 == v2
        if not same:
            changed += 1
        marker = " " if same else "≠"
        v1_s = str(v1)[:34]
        v2_s = str(v2)[:34]
        typer.echo(f"  {marker} {label:<22} {v1_s:<36} {v2_s:<36}")

    # Narrative diff
    n1 = bool(pack1.narrative_md)
    n2 = bool(pack2.narrative_md)
    n_same = n1 == n2
    if not n_same:
        changed += 1
    marker = " " if n_same else "≠"
    typer.echo(f"  {marker} {'Narrative':<22} {'yes' if n1 else 'no':<36} {'yes' if n2 else 'no':<36}")

    # Handoff diff
    h1 = bool(pack1.handoff_md)
    h2 = bool(pack2.handoff_md)
    h_same = h1 == h2
    if not h_same:
        changed += 1
    marker = " " if h_same else "≠"
    typer.echo(f"  {marker} {'Handoff':<22} {'yes' if h1 else 'no':<36} {'yes' if h2 else 'no':<36}")

    # Tags diff
    t1 = set(pack1.manifest.tags)
    t2 = set(pack2.manifest.tags)
    if t1 != t2:
        changed += 1
        added = t2 - t1
        removed = t1 - t2
        typer.echo(f"  {'≠ Tags added':<24} {'':36} {', '.join(added) if added else '—'}")
        typer.echo(f"  {'≠ Tags removed':<24} {', '.join(removed) if removed else '—'}")
        if t1:
            typer.echo(f"  {'  common':<24} {', '.join(sorted(t1 & t2))}")

    if verbose and pack1.narrative_md and pack2.narrative_md:
        if pack1.narrative_md != pack2.narrative_md:
            typer.echo("")
            typer.echo("── Narrative diff ──")
            try:
                import difflib
                for line in difflib.unified_diff(
                    pack1.narrative_md.splitlines(),
                    pack2.narrative_md.splitlines(),
                    fromfile=pa.name,
                    tofile=pb.name,
                    lineterm="",
                ):
                    typer.echo(line)
            except ImportError:
                typer.echo("(difflib not available)")

    typer.echo("")
    if changed == 0:
        typer.echo("✅ Packs are identical")
    else:
        typer.echo(f"{'≠'} {changed} field(s) differ")


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

    heading = f'Recent sessions matching "{search}"' if search else "Recent sessions"
    typer.echo(f"📋 {heading}\n")
    for s in sessions:
        sid = s["id"]
        title = s.get("title") or "(untitled)"
        source = s.get("source", "?")
        model = s.get("model", "?")
        msgs = s.get("message_count", 0)
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
    narrate: bool = typer.Option(False, "--narrate", "-n", help="Compress narrative with LLM (requires Cloudflare Workers AI credentials)"),
    deep: bool = typer.Option(False, "--deep", help="Use deep reasoning model (GLM-5.2, slower but richer). Implies --narrate."),
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
            narrate=narrate or deep,
            deep=deep,
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
    if narrate or deep:
        typer.echo(f"   Narrated: {'✅' if narrate or deep else '❌'}")


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


# ── search: grep across packs ──────────────────────────────────────


@app.command()
def search(
    pattern: str = typer.Argument(..., help="Regex pattern to search for"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Directory to scan for packs"),
    case_insensitive: bool = typer.Option(False, "-i", "--ignore-case", help="Case-insensitive search"),
    context_lines: int = typer.Option(0, "--context", "-C", help="Lines of context around each match"),
):
    """Search across .hermes-memory packs for a pattern.

    Scans manifest.json, narrative.md, decisions.json, messages.json,
    and context/*.md in every pack found in the given directory.
    """
    packs = sorted(Path(path).glob(f"*{HERMES_MEMORY_EXTENSION}"))
    if not packs:
        typer.echo(f"No .hermes-memory packs found in {path}")
        raise typer.Exit(1)

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        typer.echo(f"❌ Invalid regex: {e}", err=True)
        raise typer.Exit(1)

    total_matches = 0
    found_any = False

    for pack_path in packs:
        try:
            pack_matches = _search_pack(pack_path, compiled, context_lines)
        except Exception:
            continue

        if pack_matches:
            found_any = True
            typer.echo(f"\n📦 {pack_path.name}")
            typer.echo("─" * 50)
            for source, lines in pack_matches:
                for line in lines:
                    typer.echo(f"  [{source}] {line}")
                typer.echo("")
            total_matches += len(pack_matches)

    if not found_any:
        typer.echo(f'No matches for "{pattern}" in {len(packs)} pack(s).')
        raise typer.Exit()

    typer.echo(f"Found {total_matches} match(es) across {len(packs)} pack(s).")


def _search_pack(
    pack_path: Path,
    pattern: re.Pattern,
    context_lines: int = 0,
) -> list[tuple[str, list[str]]]:
    """Search inside a single .hermes-memory pack and return matches.

    Returns list of (source_filename, [matching_lines]).
    """
    results: list[tuple[str, list[str]]] = []
    search_targets = [
        "manifest.json",
        "narrative.md",
        "messages.json",
        "decisions.json",
        "tool-traces.json",
        "context/handoff.md",
        "context/references.md",
    ]

    with tarfile.open(pack_path, "r:*") as tar:
        members = {m.name: m for m in tar.getmembers()}

        for target in search_targets:
            if target not in members:
                continue

            try:
                content = tar.extractfile(target).read().decode("utf-8", errors="replace")
            except Exception:
                continue

            lines = content.split("\n")
            matched_lines: list[str] = []

            for i, line in enumerate(lines):
                if pattern.search(line):
                    # Build context window
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    ctx_text = lines[start:end]

                    # Indicate which line matched
                    if context_lines > 0:
                        annotated = []
                        for j, ctx_line in enumerate(ctx_text):
                            line_no = start + j + 1
                            marker = "→" if start + j == i else " "
                            annotated.append(f"  {marker} L{line_no}: {ctx_line}")
                        matched_lines.extend(annotated)
                        matched_lines.append("")  # separator between matches
                    else:
                        matched_lines.append(f"L{i + 1}: {line}")

            if matched_lines:
                results.append((target, matched_lines))

    return results


# ── render: export pack to human-readable format ────────────────────


@app.command()
def render(
    path: Path = typer.Argument(..., help="Path to .hermes-memory file", exists=True),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown or html"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Render a context pack as a human-readable document (markdown or HTML)."""
    pack = ContextPack.read(path)

    if format == "markdown":
        from memory_vault.core.renderer import render_markdown
        content = render_markdown(pack)
        suffix = ".md"
    elif format == "html":
        from memory_vault.core.renderer import render_html
        content = render_html(pack)
        suffix = ".html"
    else:
        typer.echo(f"❌ Unknown format: {format}. Use markdown or html.", err=True)
        raise typer.Exit(1)

    if output:
        out_path = Path(output)
    else:
        stem = path.stem.replace(HERMES_MEMORY_EXTENSION, "") if path.stem.endswith(HERMES_MEMORY_EXTENSION) else path.stem
        out_path = Path.cwd() / f"{stem}{suffix}"

    out_path.write_text(content, encoding="utf-8")
    typer.echo(f"✅ Rendered → {out_path}")

    # Show preview (first 30 lines)
    preview_lines = content.split("\n")[:30]
    typer.echo(f"   Preview ({format}, {len(content)} chars):")
    for line in preview_lines:
        typer.echo(f"   {line}")


# ── index: enrich session titles & summaries with LLM ──────────────


@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Re-index already-indexed sessions"),
    new_only: bool = typer.Option(False, "--new", "-n", help="Only index new sessions (skip indexed)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model override (default: MEMORY_VAULT_INDEX_MODEL env or llama-3.3-70b)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers (1 = sequential)"),
    hermes_home: Optional[str] = typer.Option(None, "--hermes-home", help="Custom Hermes home path"),
):
    """Scan Hermes sessions and generate descriptive titles + summaries via LLM.

    Stores results in a local SQLite index so the TUI shows meaningful
    session names instead of auto-generated ones.

    Use --workers 4 to index multiple sessions in parallel (faster on large vaults).

    To configure the LLM model:
      export MEMORY_VAULT_INDEX_MODEL="@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in environment.
    Falls back to template-based labels when Cloudflare is unavailable.
    """
    from memory_vault.core.builder import ContextBuilder
    from memory_vault.core.session_index import SessionIndex

    builder = ContextBuilder(hermes_home=hermes_home)
    idx = SessionIndex()

    # Check LLM provider availability
    if not idx._provider.available():
        typer.echo("⚠️  LLM provider credentials not found — will use template fallback titles.")
        typer.echo("   Configure via: memory-vault config set llm.provider <name>")
        typer.echo("   Or set MEMORY_VAULT_LLM_PROVIDER=<name> in your environment.")
        typer.echo("")

    if model:
        idx.set_model(model)
        typer.echo(f"   Using model: {model}")

    if workers > 1:
        idx.set_workers(workers)
        typer.echo(f"   Parallel workers: {workers}")

    # Show current index status
    stats_before = idx.summary_stats()
    typer.echo(f"📊 Index before: {stats_before['total']} sessions ({stats_before['with_summary']} with summaries)")
    typer.echo("")

    # Run indexing
    with typer.progressbar(length=1, label="Indexing sessions...") as progress:
        progress.update(1)

        if new_only:
            result = idx.index_new(builder)
        else:
            result = idx.index_all(builder, force=force)

    # Report
    stats_after = idx.summary_stats()
    typer.echo("")
    typer.echo(f"✅ Done — {result.get('indexed', 0)} indexed, {result.get('skipped', 0)} skipped, "
               f"{result.get('errors', 0)} errors")
    typer.echo(f"📊 Index now: {stats_after['total']} sessions ({stats_after['with_summary']} with summaries)")

    if result.get("error"):
        typer.echo(f"⚠️  {result['error']}")


# ── browse: interactive TUI ─────────────────────────────────────────


@app.command()
def browse(
    mode: str = typer.Option("sessions", "--mode", "-m", help="Browse mode: sessions or packs"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Directory for pack browsing"),
):
    """Launch interactive TUI to browse sessions or packs.

    Requires the [tui] extra: pip install 'memory-vault[tui]'
    """
    try:
        from memory_vault.core.tui import browse as tui_browse
        tui_browse(mode=mode, directory=path)
    except ImportError:
        typer.echo("❌ TUI extra not installed. Run: pip install 'memory-vault[tui]'", err=True)
        raise typer.Exit(1)


def main():
    app()
