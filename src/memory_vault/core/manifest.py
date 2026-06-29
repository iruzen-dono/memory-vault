"""Manifest: metadata for a .hermes-memory context pack.

Every pack carries a manifest.json that describes the context,
its origin session, and what's inside.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

CURRENT_FORMAT_VERSION = "1.0.0"
PACK_TYPE_CONTEXT = "context-pack"
PACK_TYPE_MEMORY = "memory-pack"  # legacy


@dataclass
class ToolUsage:
    """How many times each tool was called during the session."""
    total_calls: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)


@dataclass
class ArtifactIndex:
    """Files created or modified during the action."""
    count: int = 0
    files: list[str] = field(default_factory=list)


@dataclass
class NarrativeIndex:
    """Structure of the narrative."""
    chapters: list[dict] = field(default_factory=list)


@dataclass
class Manifest:
    """Pack metadata — serialised to manifest.json."""

    format_version: str = CURRENT_FORMAT_VERSION
    pack_type: str = PACK_TYPE_CONTEXT
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    author: str = ""

    # Source session info
    source_session_id: str = ""
    source_platform: str = ""           # cli, telegram, discord, …
    source_model: str = ""              # model used during the session
    duration_minutes: int = 0
    message_count: int = 0

    # Contents summary
    narrative: NarrativeIndex = field(default_factory=NarrativeIndex)
    artifacts: ArtifactIndex = field(default_factory=ArtifactIndex)
    tool_usage: ToolUsage = field(default_factory=ToolUsage)

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        narrative_data = data.pop("narrative", {})
        artifacts_data = data.pop("artifacts", {})
        tool_usage_data = data.pop("tool_usage", {})

        narrative = NarrativeIndex(**narrative_data)
        artifacts = ArtifactIndex(**artifacts_data)
        tool_usage = ToolUsage(**tool_usage_data)

        return cls(**data, narrative=narrative, artifacts=artifacts, tool_usage=tool_usage)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Manifest":
        return cls.from_dict(json.loads(raw))

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.title:
            errors.append("title is required")
        if self.pack_type not in (PACK_TYPE_CONTEXT, PACK_TYPE_MEMORY):
            errors.append(f"unknown pack type: {self.pack_type}")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0
