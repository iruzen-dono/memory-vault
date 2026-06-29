"""Memory Vault — Portable Context Protocol.

A standard format + CLI for packaging AI agent sessions into portable
.hermes-memory context packs: conversations, tool traces, artifacts,
decisions, and a handoff brief for another agent.
"""

from memory_vault.core.builder import ContextBuilder
from memory_vault.core.manifest import (
    CURRENT_FORMAT_VERSION,
    PACK_TYPE_CONTEXT,
    ArtifactIndex,
    Manifest,
    ToolUsage,
)
from memory_vault.core.pack import ContextPack
from memory_vault.core.session_index import SessionIndex

__version__ = "0.1.0"
__all__ = [
    "ContextPack",
    "ContextBuilder",
    "Manifest",
    "ToolUsage",
    "ArtifactIndex",
    "SessionIndex",
    "CURRENT_FORMAT_VERSION",
    "PACK_TYPE_CONTEXT",
]
