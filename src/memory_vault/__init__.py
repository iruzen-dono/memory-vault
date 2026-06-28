"""Memory Vault — Portable Context Protocol.

A standard format + CLI for packaging AI agent sessions into portable
.hermes-memory context packs: conversations, tool traces, artifacts,
decisions, and a handoff brief for another agent.
"""

from memory_vault.core.pack import ContextPack
from memory_vault.core.manifest import Manifest, ToolUsage, ArtifactIndex, CURRENT_FORMAT_VERSION, PACK_TYPE_CONTEXT
from memory_vault.core.builder import ContextBuilder

__version__ = "0.1.0"
__all__ = [
    "ContextPack",
    "ContextBuilder",
    "Manifest",
    "ToolUsage",
    "ArtifactIndex",
    "CURRENT_FORMAT_VERSION",
    "PACK_TYPE_CONTEXT",
]
