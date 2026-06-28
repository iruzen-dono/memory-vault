"""ContextPack — the core abstraction for .hermes-memory context packs.

A context pack is a tar.gz archive that captures a complete action or
event from an AI agent session: the conversation, decisions, artifacts,
tool usage, and a handoff brief for another agent to pick up where
the session left off.

Layout (v1 — context-pack):

    manifest.json          — metadata (pack_type="context-pack")
    narrative.md           — chronological story of the action
    messages.json          — raw session messages for reference
    decisions.json         — key decisions extracted (optional)
    artifacts/             — files created or modified during the action
    tool-traces.json       — tool usage summary
    context/
        handoff.md         — brief for the receiving agent
        references.md      — links, docs, resources used
"""

from __future__ import annotations

import json
import tarfile
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .manifest import Manifest, ToolUsage, ArtifactIndex, NarrativeIndex


HERMES_MEMORY_EXTENSION = ".hermes-memory"


def _ensure_lf(text: str) -> str:
    """Normalize line endings to LF (Unix style)."""
    return text.replace("\r\n", "\n")


@dataclass
class ContextPack:
    """In-memory representation of a .hermes-memory context pack.

    The core unit of the protocol: captures an action/event with its
    full context — conversation, decisions, artifacts, tool traces,
    and a handoff brief for another agent.
    """

    manifest: Manifest
    narrative_md: str = ""
    messages: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)  # path_in_pack -> source file
    tool_traces: dict = field(default_factory=dict)
    handoff_md: str = ""
    references_md: str = ""

    # -- Serialisation ------------------------------------------------

    def write(self, path: str | Path, compress: bool = True) -> Path:
        """Write the pack to a .hermes-memory file (tar.gz).

        Uses a staging directory for reliable multi-file creation.
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(HERMES_MEMORY_EXTENSION)

        with tempfile.TemporaryDirectory(prefix="ctxpack_") as tmp:
            stage = Path(tmp)

            # 1. Manifest
            (stage / "manifest.json").write_text(
                json.dumps(self.manifest.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # 2. Narrative
            if self.narrative_md:
                (stage / "narrative.md").write_bytes(
                    _ensure_lf(self.narrative_md).encode("utf-8")
                )

            # 3. Messages (raw)
            if self.messages:
                (stage / "messages.json").write_text(
                    json.dumps(self.messages, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            # 4. Decisions
            if self.decisions:
                (stage / "decisions.json").write_text(
                    json.dumps(self.decisions, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            # 5. Artifacts — copy files into staged layout
            if self.artifacts:
                artifacts_dir = stage / "artifacts"
                for dest_rel, source_path in self.artifacts.items():
                    dest = artifacts_dir / dest_rel
                    if source_path.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, dest)

            # 6. Tool traces
            if self.tool_traces:
                (stage / "tool-traces.json").write_text(
                    json.dumps(self.tool_traces, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            # 7. Context (handoff + references)
            if self.handoff_md or self.references_md:
                ctx_dir = stage / "context"
                ctx_dir.mkdir(exist_ok=True)
                if self.handoff_md:
                    (ctx_dir / "handoff.md").write_bytes(
                        _ensure_lf(self.handoff_md).encode("utf-8")
                    )
                if self.references_md:
                    (ctx_dir / "references.md").write_bytes(
                        _ensure_lf(self.references_md).encode("utf-8")
                    )

            # Create tar.gz from staging directory
            mode = "w:gz" if compress else "w"
            with tarfile.open(path, mode) as tar:
                for item in stage.iterdir():
                    tar.add(item, arcname=item.name, recursive=True)

        return path.resolve()

    @classmethod
    def read(cls, path: str | Path) -> "ContextPack":
        """Read a .hermes-memory file into memory."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pack not found: {path}")

        with tarfile.open(path, "r:*") as tar:
            members = {m.name: m for m in tar.getmembers()}

            # Manifest is required
            if "manifest.json" not in members:
                raise ValueError(f"Invalid pack: missing manifest.json in {path}")

            manifest_data = json.loads(tar.extractfile("manifest.json").read())
            manifest = Manifest.from_dict(manifest_data)

            # Narrative
            narrative_md = ""
            if "narrative.md" in members:
                narrative_md = tar.extractfile("narrative.md").read().decode("utf-8")

            # Messages
            messages = []
            if "messages.json" in members:
                messages = json.loads(tar.extractfile("messages.json").read())

            # Decisions
            decisions = []
            if "decisions.json" in members:
                decisions = json.loads(tar.extractfile("decisions.json").read())

            # Tool traces
            tool_traces = {}
            if "tool-traces.json" in members:
                tool_traces = json.loads(tar.extractfile("tool-traces.json").read())

            # Artifacts — extract to a single temp dir
            artifacts: dict[str, Path] = {}
            artifact_members = [
                m for m in tar.getmembers()
                if m.name.startswith("artifacts/") and m.isfile()
            ]
            if artifact_members:
                artifact_root = Path(tempfile.mkdtemp(prefix="ctxart_"))
                for m in artifact_members:
                    # Get the relative path within artifacts/ (POSIX-style)
                    rel_path = str(Path(m.name).relative_to("artifacts")).replace("\\", "/")
                    dest = artifact_root / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(m).read())
                    artifacts[rel_path] = dest

            # Context
            handoff_md = ""
            references_md = ""
            if "context/handoff.md" in members:
                handoff_md = tar.extractfile("context/handoff.md").read().decode("utf-8")
            if "context/references.md" in members:
                references_md = tar.extractfile("context/references.md").read().decode("utf-8")

        return cls(
            manifest=manifest,
            narrative_md=narrative_md,
            messages=messages,
            decisions=decisions,
            artifacts=artifacts,
            tool_traces=tool_traces,
            handoff_md=handoff_md,
            references_md=references_md,
        )

    # -- Helpers ------------------------------------------------------

    def summary(self) -> dict:
        """Human-readable summary of the pack."""
        artifact_files = list(self.artifacts.keys())
        return {
            "title": self.manifest.title,
            "description": self.manifest.description,
            "author": self.manifest.author,
            "tags": self.manifest.tags,
            "pack_type": self.manifest.pack_type,
            "source_session_id": self.manifest.source_session_id,
            "message_count": len(self.messages),
            "artifact_count": len(self.artifacts),
            "artifact_files": artifact_files,
            "decisions": len(self.decisions),
            "has_narrative": bool(self.narrative_md),
            "has_handoff": bool(self.handoff_md),
            "format_version": self.manifest.format_version,
            "created_at": self.manifest.created_at,
        }
