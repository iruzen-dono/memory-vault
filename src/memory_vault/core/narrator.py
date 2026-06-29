"""Narrator — AI session compression for context packs.

Uses a pluggable LLM provider (Cloudflare Workers AI, OpenAI-compatible,
or template fallback) to generate a concise narrative, extract key
decisions, and produce a handoff brief from a Hermes session.

Provider selection (see core/llm.py):
  - MEMORY_VAULT_LLM_PROVIDER env var
  - Auto-detect from credentials
  - Template fallback if no provider available
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from memory_vault.core.llm import (
    DEEP_MODEL,
    FAST_MODEL,
    LLMProvider,
    get_provider,
)

# ── Data classes ────────────────────────────────────────────────────


@dataclass
class NarrativeResult:
    """Result of LLM-powered session compression."""

    summary_md: str = ""
    """Concise executive summary (3-8 bullet points)."""

    decisions: list[dict] = field(default_factory=list)
    """List of {what, why, by_tool} extracted decisions."""

    handoff_md: str = ""
    """Handoff brief for another agent to continue the work."""

    model_used: str = ""
    """Model name used for compression."""

    compressed: bool = False
    """True if LLM was actually called (not template fallback)."""


# ── The narrator ────────────────────────────────────────────────────


class SessionNarrator:
    """Compresses a session into narrative + decisions + handoff.

    Usage::

        narrator = SessionNarrator()
        result = narrator.summarize(session, messages, tool_traces)

    Args:
        provider: An LLMProvider instance. Defaults to auto-detect.
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or get_provider()

    def summarize(
        self,
        session: dict,
        messages: list[dict],
        tool_traces: dict | None = None,
        deep: bool = False,
    ) -> NarrativeResult:
        """Generate a compressed narrative from session data.

        Tries the LLM path first (via the configured provider).
        Falls back to template-based summarization.
        """
        # Try LLM path
        if self.provider.available():
            prompt = self._build_prompt(session, messages, tool_traces)
            model = DEEP_MODEL if deep else FAST_MODEL
            # Cloudflare uses CF model names; OpenAI uses its own
            if self.provider.name != "cloudflare":
                model = self.provider.default_model() if deep else self.provider.default_model()
            response = self.provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise session summarizer. "
                                   "Produce clear, structured markdown. Be concise.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=8192 if deep else 4096,
                temperature=0.3,
            )
            if response:
                return self._parse_llm_response(response, model=model)

        # Fallback: template
        return self._template_fallback(session, messages, tool_traces)

    # -- Prompt building ------------------------------------------------

    def _build_prompt(
        self,
        session: dict,
        messages: list[dict],
        tool_traces: dict | None = None,
    ) -> str:
        """Build the LLM prompt from session data."""
        title = session.get("title") or "Untitled Session"
        model = session.get("model", "unknown")
        platform = session.get("source", "unknown")
        msg_count = len(messages)
        total_tool_calls = (tool_traces or {}).get("total_tool_calls", 0)

        # Sample messages (trim for context window)
        sample_lines = []
        for i, msg in enumerate(messages[:150]):  # cap at 150 messages
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:500]
            tool_calls = msg.get("tool_calls")
            tool_name = msg.get("tool_name", "")

            if role == "user":
                sample_lines.append(f"## USER ({i})\n{content}\n")
            elif role == "assistant":
                if content:
                    sample_lines.append(f"## ASSISTANT ({i})\n{content}\n")
                if tool_calls:
                    try:
                        tc = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                        for t in (tc if isinstance(tc, list) else [tc]):
                            name = t.get("name", t.get("function", {}).get("name", "tool"))
                            sample_lines.append(f"[TOOL CALL: {name}]\n")
                    except (json.JSONDecodeError, TypeError):
                        pass
            elif role == "tool":
                short = (content or "")[:200]
                sample_lines.append(f"[TOOL RESULT: {tool_name}] {short}\n")

        conversation_sample = "\n".join(sample_lines)

        # Tool usage summary
        tool_summary = ""
        if tool_traces:
            by_tool = tool_traces.get("by_tool", {})
            tool_summary = "\n".join(
                f"  - {name}: {count}x" for name, count in by_tool.items()
            )

        return f"""\
# Session to Summarize

**Title:** {title}
**Model:** {model}
**Platform:** {platform}
**Messages:** {msg_count} | **Tool calls:** {total_tool_calls}

## Conversation Transcript (truncated to 150 messages)

{conversation_sample}

## Tool Usage

{tool_summary if tool_summary else "No tool calls recorded."}

---

## Your Task

Analyze the session above and produce THREE sections:

### 1. Executive Summary (3-8 bullet points)
What was accomplished? What was the goal, approach, and outcome?
Be concrete — mention files changed, tools used, and results.

### 2. Key Decisions
List every meaningful decision made during the session.
For each: what was decided, why, and which tool/task it related to.
Format as a markdown table:

| Decision | Rationale | Context |
|----------|-----------|---------|

### 3. Handoff Brief
A concise paragraph (2-5 sentences) telling another agent:
- What was the goal
- What was completed
- What remains / next steps
- Any gotchas or important context to know

Write in natural language, as if briefing a teammate who is
taking over this task right now."""

    # -- Response parsing -----------------------------------------------

    def _parse_llm_response(self, response: str, model: str = "unknown") -> NarrativeResult:
        """Parse the LLM's response into structured sections."""
        # Extract sections by markdown headings
        sections = {}
        current_section = "preamble"
        current_lines = []

        for line in response.split("\n"):
            heading_match = re.match(r"^###?\s+(.+)", line.strip())
            if heading_match:
                sections[current_section] = "\n".join(current_lines).strip()
                current_section = heading_match.group(1).lower().strip()
                current_lines = []
            else:
                current_lines.append(line)

        sections[current_section] = "\n".join(current_lines).strip()

        # Map known section headings
        summary_md = (
            sections.get("executive summary")
            or sections.get("1. executive summary")
            or sections.get("summary")
            or ""
        )
        handoff_md = (
            sections.get("handoff brief")
            or sections.get("3. handoff brief")
            or sections.get("handoff")
            or ""
        )

        # Extract decisions from the key decisions table
        decisions = []
        decisions_text = (
            sections.get("key decisions")
            or sections.get("2. key decisions")
            or ""
        )

        # Try to parse markdown table rows
        table_pattern = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|")
        for line in decisions_text.split("\n"):
            m = table_pattern.match(line.strip())
            if m and not line.startswith("|---"):
                decision = m.group(1).strip()
                rationale = m.group(2).strip()
                context = m.group(3).strip()
                if decision and decision.lower() not in ("decision", ""):
                    decisions.append({
                        "what": decision,
                        "why": rationale,
                        "context": context,
                    })

        # Build combined summary
        full_summary = "# Compressed Narrative\n\n"
        if summary_md:
            full_summary += summary_md + "\n\n"
        if decisions:
            full_summary += "## Key Decisions\n\n"
            full_summary += "| Decision | Rationale | Context |\n"
            full_summary += "|----------|-----------|--------|\n"
            for d in decisions:
                full_summary += f"| {d['what']} | {d['why']} | {d['context']} |\n"
            full_summary += "\n"

        return NarrativeResult(
            summary_md=full_summary,
            decisions=decisions,
            handoff_md=handoff_md,
            model_used=model,
            compressed=True,
        )

    # -- Template fallback ----------------------------------------------

    def _template_fallback(
        self,
        session: dict,
        messages: list[dict],
        tool_traces: dict | None = None,
    ) -> NarrativeResult:
        """Template-based summarization without LLM."""
        title = session.get("title") or "Untitled"
        model_name = session.get("model", "unknown")
        msg_count = len(messages)
        total_tool_calls = (tool_traces or {}).get("total_tool_calls", 0)

        # Find first user message (goal)
        goal = ""
        for msg in messages:
            if msg.get("role") == "user":
                goal = (msg.get("content") or "")[:300]
                break

        # Find last assistant message (conclusion)
        conclusion = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    conclusion = content[:300]
                    break

        summary_lines = [
            "# Compressed Narrative\n",
            f"**Session:** {title}",
            f"**Model:** {model_name}",
            f"**Messages:** {msg_count} | **Tool calls:** {total_tool_calls}",
            "",
            "## Summary",
            "",
        ]
        if goal:
            summary_lines.append(f"**Goal:** {goal}")
            summary_lines.append("")
        if conclusion:
            summary_lines.append(f"**Outcome:** {conclusion}")
            summary_lines.append("")

        handoff_lines = [
            "# Handoff Brief",
            "",
            f"Goal: {goal}" if goal else "",
            "",
            "Key decisions and next steps are embedded in the full narrative.",
            "Load the session messages and tool traces for complete context.",
            "",
        ]

        return NarrativeResult(
            summary_md="\n".join(summary_lines),
            handoff_md="\n".join(handoff_lines),
            model_used="template",
            compressed=False,
        )
