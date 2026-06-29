"""Renderer — export context packs to human-readable formats.

Supports:
  - Markdown (readable transcript with narrative + decisions + messages)
  - HTML (standalone, self-contained with CSS styling)

Usage:
    from memory_vault.core.renderer import render_markdown, render_html
    md = render_markdown(pack)
    html = render_html(pack)
"""

from __future__ import annotations

import html as html_mod
from datetime import datetime

from .pack import ContextPack


# ── Markdown ────────────────────────────────────────────────────────


def render_markdown(pack: ContextPack) -> str:
    """Render a context pack as a human-readable markdown document."""
    lines: list[str] = []
    s = pack.summary()

    # Title & metadata
    lines.append(f"# {s['title']}")
    lines.append("")
    if s["description"]:
        lines.append(f"_{s['description']}_")
        lines.append("")

    lines.append("## Metadata")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Pack type | {s['pack_type']} |")
    lines.append(f"| Format version | {s['format_version']} |")
    lines.append(f"| Author | {s['author'] or '—'} |")
    lines.append(f"| Created | {s['created_at']} |")
    lines.append(f"| Tags | {', '.join(s['tags']) or '—'} |")
    if s["source_session_id"]:
        lines.append(f"| Source session | `{s['source_session_id']}` |")
    lines.append(f"| Messages | {s['message_count']} |")
    lines.append(f"| Artifacts | {s['artifact_count']} files |")
    lines.append(f"| Decisions | {s['decisions']} |")
    lines.append("")

    # Narrative
    if pack.narrative_md:
        narrative = pack.narrative_md.strip()
        # If it already has headings, embed it; otherwise wrap
        if narrative.startswith("#"):
            lines.append(narrative)
        else:
            lines.append("## Narrative")
            lines.append("")
            lines.append(narrative)
        lines.append("")

    # Decisions
    if pack.decisions:
        lines.append("## Key Decisions")
        lines.append("")
        lines.append("| # | Decision | Rationale | Context |")
        lines.append("|---|----------|-----------|---------|")
        for i, d in enumerate(pack.decisions, 1):
            what = d.get("what", d.get("decision", ""))
            why = d.get("why", d.get("rationale", ""))
            ctx = d.get("context", "")
            lines.append(f"| {i} | {what} | {why} | {ctx} |")
        lines.append("")

    # Handoff
    if pack.handoff_md:
        lines.append("## Handoff Brief")
        lines.append("")
        lines.append(pack.handoff_md.strip())
        lines.append("")

    # Tool traces
    if pack.tool_traces:
        lines.append("## Tool Usage")
        lines.append("")
        by_tool = pack.tool_traces.get("by_tool", {})
        total = pack.tool_traces.get("total_tool_calls", 0)
        lines.append(f"_Total tool calls: {total}_")
        lines.append("")
        lines.append("| Tool | Calls |")
        lines.append("|------|-------|")
        for tool_name, count in sorted(by_tool.items(), key=lambda x: -x[1]):
            lines.append(f"| {tool_name} | {count} |")
        lines.append("")

    # Artifacts
    if pack.artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for fpath in sorted(pack.artifacts.keys()):
            lines.append(f"- `{fpath}`")
        lines.append("")

    # Messages
    if pack.messages:
        lines.append("## Messages")
        lines.append("")
        for msg in pack.messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "").strip()
            tool_calls_raw = msg.get("tool_calls", "")
            tool_name = msg.get("tool_name", "")

            if role == "user":
                lines.append("### 🧑 User")
                lines.append("")
                if content:
                    lines.append(content)
                    lines.append("")
            elif role == "assistant":
                lines.append("### 🤖 Assistant")
                lines.append("")
                if content:
                    lines.append(content)
                    lines.append("")
                if tool_calls_raw:
                    try:
                        import json
                        tc_list = json.loads(tool_calls_raw) if isinstance(tool_calls_raw, str) else (tool_calls_raw if isinstance(tool_calls_raw, list) else [tool_calls_raw])
                        for tc in tc_list:
                            tname = tc.get("name", tc.get("function", {}).get("name", "tool"))
                            lines.append(f"_→ Tool call: `{tname}`_")
                            lines.append("")
                    except (json.JSONDecodeError, TypeError):
                        pass
            elif role == "tool":
                brief = (content[:200] + "…") if len(content) > 200 else content
                lines.append(f"### 🛠 Tool Result (`{tool_name}`)")
                lines.append("")
                lines.append(f"```\n{brief}\n```")
                lines.append("")

    # References
    if pack.references_md:
        lines.append("## References")
        lines.append("")
        lines.append(pack.references_md.strip())
        lines.append("")

    return "\n".join(lines)


# ── HTML ────────────────────────────────────────────────────────────

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 15px; line-height: 1.6; color: #1a1a2e; background: #f8f9fc; max-width: 960px; margin: 0 auto; padding: 40px 20px; }
h1, h2, h3, h4 { color: #16213e; margin-top: 1.5em; margin-bottom: 0.5em; font-weight: 600; }
h1 { font-size: 1.8em; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }
h2 { font-size: 1.3em; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }
h3 { font-size: 1.1em; color: #0f3460; }
p { margin: 0.5em 0; }
table { width: 100%; border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left; }
th { background: #0f3460; color: white; font-weight: 600; }
tr:nth-child(even) { background: #f1f3f8; }
code { background: #e8eaf6; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; font-family: 'JetBrains Mono', 'Fira Code', monospace; }
pre { background: #1a1a2e; color: #e8eaf6; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.85em; margin: 1em 0; }
blockquote { border-left: 4px solid #0f3460; padding: 8px 16px; margin: 1em 0; background: #f1f3f8; }
.meta { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 1em 0; }
.meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.meta-item { display: flex; gap: 8px; }
.meta-label { font-weight: 600; color: #0f3460; min-width: 100px; }
.artifact-list { list-style: none; }
.artifact-list li { padding: 4px 0; }
.role-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; margin-bottom: 4px; }
.role-user { background: #e3f2fd; color: #1565c0; }
.role-assistant { background: #f3e5f5; color: #7b1fa2; }
.role-tool { background: #fff3e0; color: #e65100; }
.msg { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; margin: 0.8em 0; }
.tool-call { font-style: italic; color: #666; font-size: 0.9em; }
.badge { display: inline-block; background: #0f3460; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
.footer { margin-top: 3em; text-align: center; color: #999; font-size: 0.85em; border-top: 1px solid #e0e0e0; padding-top: 1em; }
"""


def _md_to_html(md_text: str) -> str:
    """Simple markdown-to-HTML converter for common patterns.

    Not a full spec implementation — handles the patterns we emit.
    """
    lines = md_text.split("\n")
    html_lines: list[str] = []
    in_code_block = False
    in_table = False

    for line in lines:
        # Fenced code blocks
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre>\n")
                in_code_block = False
            else:
                lang = line[3:].strip()
                html_lines.append(f"<pre><code>" if not lang else f'<pre><code class="language-{html_mod.escape(lang)}">')
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(html_mod.escape(line) + "\n")
            continue

        stripped = line.strip()

        # Horizontal rules
        if stripped == "---":
            html_lines.append("<hr>\n")
            continue

        # Headings
        if stripped.startswith("### "):
            html_lines.append(f"<h3>{_inline_md(stripped[4:])}</h3>\n")
            continue
        if stripped.startswith("## "):
            html_lines.append(f"<h2>{_inline_md(stripped[3:])}</h2>\n")
            continue
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{_inline_md(stripped[2:])}</h1>\n")
            continue

        # Tables
        if "|" in stripped and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(c == "---" or c == "---" or "---" in c for c in cells) and len(cells) > 1:
                # separator row — skip
                continue
            if not in_table:
                html_lines.append("<table>\n")
                in_table = True
            # Check if it's a header row (first row after separator)
            is_header = not in_table
            tag = "th" if is_header else "td"
            html_lines.append("<tr>" + "".join(f"<{tag}>{_inline_md(c)}</{tag}>" for c in cells) + "</tr>\n")
            continue
        if in_table and not stripped.startswith("|"):
            html_lines.append("</table>\n")
            in_table = False

        # Empty lines
        if not stripped:
            html_lines.append("<br>\n")
            continue

        # Blockquotes
        if stripped.startswith("> "):
            html_lines.append(f"<blockquote>{_inline_md(stripped[2:])}</blockquote>\n")
            continue

        # Regular paragraph
        html_lines.append(f"<p>{_inline_md(stripped)}</p>\n")

    if in_code_block:
        html_lines.append("</pre>\n")
    if in_table:
        html_lines.append("</table>\n")

    return "".join(html_lines)


def _inline_md(text: str) -> str:
    """Process inline markdown patterns."""
    escaped = html_mod.escape(text)
    # Code: `text`
    escaped = __import__("re").sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    # Bold: **text**
    escaped = __import__("re").sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    # Italic: _text_ or *text*
    escaped = __import__("re").sub(r"_([^_]+)_", r"<em>\1</em>", escaped)
    escaped = __import__("re").sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    # Links
    escaped = __import__("re").sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def render_html(pack: ContextPack, title: str | None = None) -> str:
    """Render a context pack as a standalone HTML document."""
    s = pack.summary()
    page_title = title or s["title"] or "Context Pack"

    body_parts: list[str] = []

    # Header
    body_parts.append(f"<h1>{html_mod.escape(s['title'])}</h1>")
    if s["description"]:
        body_parts.append(f"<p><em>{html_mod.escape(s['description'])}</em></p>")

    # Metadata
    body_parts.append("<div class='meta'>")
    body_parts.append("<div class='meta-grid'>")
    meta_items = [
        ("Type", s["pack_type"]),
        ("Version", s["format_version"]),
        ("Author", s["author"] or "—"),
        ("Created", s["created_at"]),
        ("Tags", ", ".join(s["tags"]) or "—"),
        ("Messages", str(s["message_count"])),
        ("Artifacts", f"{s['artifact_count']} files"),
        ("Decisions", str(s["decisions"])),
    ]
    for label, value in meta_items:
        body_parts.append(f"<div class='meta-item'><span class='meta-label'>{label}:</span><span>{html_mod.escape(str(value))}</span></div>")
    body_parts.append("</div></div>")

    # Narrative
    if pack.narrative_md:
        body_parts.append("<h2>Narrative</h2>")
        body_parts.append(_md_to_html(pack.narrative_md))

    # Decisions
    if pack.decisions:
        body_parts.append("<h2>Key Decisions</h2>")
        body_parts.append("<table><tr><th>#</th><th>Decision</th><th>Rationale</th><th>Context</th></tr>")
        for i, d in enumerate(pack.decisions, 1):
            body_parts.append(
                f"<tr><td>{i}</td><td>{html_mod.escape(str(d.get('what', d.get('decision', ''))))}</td>"
                f"<td>{html_mod.escape(str(d.get('why', d.get('rationale', ''))))}</td>"
                f"<td>{html_mod.escape(str(d.get('context', '')))}</td></tr>"
            )
        body_parts.append("</table>")

    # Handoff
    if pack.handoff_md:
        body_parts.append("<h2>Handoff Brief</h2>")
        body_parts.append(_md_to_html(pack.handoff_md))

    # Tool traces
    if pack.tool_traces:
        body_parts.append("<h2>Tool Usage</h2>")
        by_tool = pack.tool_traces.get("by_tool", {})
        total = pack.tool_traces.get("total_tool_calls", 0)
        body_parts.append(f"<p>Total tool calls: <span class='badge'>{total}</span></p>")
        body_parts.append("<table><tr><th>Tool</th><th>Calls</th></tr>")
        for tool_name, count in sorted(by_tool.items(), key=lambda x: -x[1]):
            body_parts.append(f"<tr><td><code>{html_mod.escape(tool_name)}</code></td><td>{count}</td></tr>")
        body_parts.append("</table>")

    # Artifacts
    if pack.artifacts:
        body_parts.append("<h2>Artifacts</h2>")
        body_parts.append("<ul class='artifact-list'>")
        for fpath in sorted(pack.artifacts.keys()):
            body_parts.append(f"<li><code>{html_mod.escape(fpath)}</code></li>")
        body_parts.append("</ul>")

    # Messages
    if pack.messages:
        body_parts.append("<h2>Messages</h2>")
        for msg in pack.messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "").strip()
            tool_calls_raw = msg.get("tool_calls", "")
            tool_name = msg.get("tool_name", "")

            badge_class = {"user": "role-user", "assistant": "role-assistant", "tool": "role-tool"}.get(role, "")
            label = {"user": "User", "assistant": "Assistant", "tool": f"Tool Result ({tool_name})"}.get(role, role.title())

            body_parts.append("<div class='msg'>")
            body_parts.append(f"<span class='role-badge {badge_class}'>{label}</span>")

            if role == "tool":
                brief = (content[:500] + "…") if len(content) > 500 else content
                body_parts.append(f"<pre><code>{html_mod.escape(brief)}</code></pre>")
            else:
                if content:
                    body_parts.append(_md_to_html(content))
                if tool_calls_raw and role == "assistant":
                    try:
                        import json
                        tc_list = json.loads(tool_calls_raw) if isinstance(tool_calls_raw, str) else (tool_calls_raw if isinstance(tool_calls_raw, list) else [tool_calls_raw])
                        for tc in tc_list:
                            tname = tc.get("name", tc.get("function", {}).get("name", "tool"))
                            body_parts.append(f"<div class='tool-call'>→ Tool call: <code>{html_mod.escape(tname)}</code></div>")
                    except (json.JSONDecodeError, TypeError):
                        pass

            body_parts.append("</div>")

    # Footer
    body_parts.append(f"<div class='footer'>Generated by Memory Vault · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_mod.escape(page_title)}</title>
<style>{_CSS}</style>
</head>
<body>
{"".join(body_parts)}
</body>
</html>"""
    return html
