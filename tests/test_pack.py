"""Tests for Memory Vault — Context Pack format."""

import json
import tarfile
import os
from pathlib import Path

import pytest

from memory_vault import __version__
from memory_vault.core.manifest import (
    Manifest, ToolUsage, ArtifactIndex, NarrativeIndex,
    PACK_TYPE_CONTEXT, CURRENT_FORMAT_VERSION,
)
from memory_vault.core.pack import ContextPack, HERMES_MEMORY_EXTENSION


# ── Manifest tests ────────────────────────────────────────────────


class TestManifest:
    def test_default_manifest(self):
        m = Manifest()
        assert m.format_version == CURRENT_FORMAT_VERSION
        assert m.pack_type == PACK_TYPE_CONTEXT
        assert m.title == ""
        assert m.tool_usage.total_calls == 0
        assert m.artifacts.count == 0
        assert m.narrative.chapters == []
        assert m.is_valid() is False  # no title

    def test_minimal_valid_manifest(self):
        m = Manifest(title="Test Pack")
        errors = m.validate()
        assert len(errors) == 0
        assert m.is_valid() is True

    def test_serialize_roundtrip(self):
        m = Manifest(
            title="Test",
            description="A test pack",
            tags=["test", "demo"],
            author="tester",
            source_session_id="sess001",
            source_platform="cli",
            source_model="deepseek-v4-flash-free",
            duration_minutes=30,
            message_count=50,
            narrative=NarrativeIndex(chapters=[{"heading": "Start", "message_idx": 0}]),
            artifacts=ArtifactIndex(count=2, files=["a.py", "b.py"]),
            tool_usage=ToolUsage(total_calls=10, by_tool={"terminal": 5, "write_file": 3}),
        )
        json_str = m.to_json()
        restored = Manifest.from_json(json_str)

        assert restored.title == "Test"
        assert restored.tags == ["test", "demo"]
        assert restored.source_session_id == "sess001"
        assert restored.tool_usage.total_calls == 10
        assert restored.tool_usage.by_tool["terminal"] == 5
        assert restored.artifacts.count == 2
        assert restored.artifacts.files == ["a.py", "b.py"]
        assert restored.narrative.chapters == [{"heading": "Start", "message_idx": 0}]

    def test_custom_pack_type(self):
        m = Manifest(title="Legacy", pack_type="memory-pack")
        assert m.pack_type == "memory-pack"
        assert m.is_valid()

    def test_invalid_pack_type(self):
        m = Manifest(title="Bad", pack_type="foo-bar")
        errors = m.validate()
        assert any("unknown pack type" in e for e in errors)


# ── ContextPack read/write tests ──────────────────────────────────


class TestContextPack:
    def test_write_read_minimal(self, tmp_path):
        """Minimal pack: just a title."""
        pack = ContextPack(
            manifest=Manifest(title="Minimal Pack"),
        )
        out = pack.write(tmp_path / "minimal.hermes-memory")
        assert out.suffix == HERMES_MEMORY_EXTENSION
        assert out.exists()

        # Verify tar contents
        with tarfile.open(out, "r:*") as tar:
            member_names = [m.name for m in tar.getmembers()]
            assert "manifest.json" in member_names

        # Read back
        restored = ContextPack.read(out)
        assert restored.manifest.title == "Minimal Pack"
        assert restored.narrative_md == ""
        assert restored.messages == []

    def test_write_read_full(self, tmp_path):
        """Full pack with narrative, messages, decisions, artifacts, context."""
        # Create a test artifact file
        src_file = tmp_path / "src" / "bot.py"
        src_file.parent.mkdir()
        src_file.write_text("# trading bot\n")

        pack = ContextPack(
            manifest=Manifest(
                title="Full Pack",
                description="Full test",
                tags=["full"],
                author="test",
                source_session_id="sess001",
                source_platform="cli",
            ),
            narrative_md="# Narrative\n\nWe built a bot.\n",
            messages=[
                {"role": "user", "content": "build a bot"},
                {"role": "assistant", "content": "ok", "tool_calls": '[{"name":"terminal","arguments":"{}"}]'},
            ],
            decisions=[{"what": "use Python", "why": "fast prototyping"}],
            artifacts={"src/bot.py": src_file},
            tool_traces={"total_tool_calls": 1, "by_tool": {"terminal": 1}},
            handoff_md="# Handoff\n\nGoal: build a bot\n",
            references_md="# References\n\nhttps://example.com\n",
        )
        out = pack.write(tmp_path / "full.hermes-memory")
        assert out.exists()

        # Read back
        restored = ContextPack.read(out)
        assert restored.manifest.title == "Full Pack"
        assert restored.narrative_md == "# Narrative\n\nWe built a bot.\n"
        assert len(restored.messages) == 2
        assert restored.messages[0]["role"] == "user"
        assert len(restored.decisions) == 1
        assert restored.decisions[0]["what"] == "use Python"
        assert "src/bot.py" in restored.artifacts
        assert restored.handoff_md == "# Handoff\n\nGoal: build a bot\n"
        assert "https://example.com" in restored.references_md

    def test_artifact_files_stored(self, tmp_path):
        """Artifact files should be readable after roundtrip."""
        src = tmp_path / "module" / "code.py"
        src.parent.mkdir()
        src.write_text("print('hello')\n")

        pack = ContextPack(
            manifest=Manifest(title="Artifact Test"),
            artifacts={"module/code.py": src},
        )
        out = pack.write(tmp_path / "artifacts.hermes-memory")

        restored = ContextPack.read(out)
        assert "module/code.py" in restored.artifacts
        artifact_path = restored.artifacts["module/code.py"]
        assert artifact_path.read_text() == "print('hello')\n"

    def test_summary(self, tmp_path):
        pack = ContextPack(
            manifest=Manifest(
                title="Summary Test",
                description="Testing summary",
                tags=["a", "b"],
                author="me",
                source_session_id="sess123",
            ),
            narrative_md="...",
            handoff_md="...",
        )
        s = pack.summary()
        assert s["title"] == "Summary Test"
        assert s["message_count"] == 0
        assert s["artifact_count"] == 0
        assert s["has_narrative"] is True
        assert s["has_handoff"] is True

    def test_roundtrip_manifest_json(self, tmp_path):
        """manifest.json inside the tar should be valid and complete."""
        pack = ContextPack(
            manifest=Manifest(
                title="JSON Roundtrip",
                description="Check manifest JSON inside tar",
                tags=["json"],
                author="tester",
                source_session_id="sess_r01",
                source_platform="telegram",
                source_model="gpt-4",
                duration_minutes=15,
                message_count=10,
                tool_usage=ToolUsage(total_calls=5, by_tool={"read_file": 5}),
            ),
            messages=[{"role": "user", "content": "hi"}],
        )
        out = pack.write(tmp_path / "json-roundtrip.hermes-memory")

        with tarfile.open(out, "r:*") as tar:
            manifest_raw = json.loads(tar.extractfile("manifest.json").read())

        assert manifest_raw["title"] == "JSON Roundtrip"
        assert manifest_raw["pack_type"] == "context-pack"
        assert manifest_raw["source_session_id"] == "sess_r01"
        assert manifest_raw["source_platform"] == "telegram"
        assert manifest_raw["source_model"] == "gpt-4"
        assert manifest_raw["message_count"] == 10  # set in manifest, not derived
        assert manifest_raw["tool_usage"]["total_calls"] == 5


# ── Builder tests (mocked Hermes DB) ─────────────────────────────


class TestManifestValidation:
    def test_invalid_no_title(self):
        m = Manifest()
        assert not m.is_valid()
        assert "title is required" in m.validate()

    def test_invalid_pack_type_empty(self):
        m = Manifest(title="Test", pack_type="")
        errors = m.validate()
        assert any("unknown pack type" in e for e in errors)


# ── Error cases ──────────────────────────────────────────────────


class TestErrorCases:
    def test_read_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ContextPack.read("/nonexistent/pack.hermes-memory")

    def test_read_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.hermes-memory"
        bad.write_text("not a tar file")
        with pytest.raises(tarfile.ReadError):
            ContextPack.read(bad)

    def test_read_missing_manifest(self, tmp_path):
        import tarfile
        bad = tmp_path / "no-manifest.hermes-memory"
        with tarfile.open(bad, "w:gz") as tar:
            info = tarfile.TarInfo(name="narrative.md")
            tar.addfile(info, b"# No manifest")
        with pytest.raises(ValueError, match="missing manifest.json"):
            ContextPack.read(bad)


# ── Version compatibility ────────────────────────────────────────


class TestVersion:
    def test_current_version_exported(self):
        assert isinstance(__version__, str)
        assert __version__ != ""

    def test_version_in_manifest(self):
        m = Manifest(title="versioned")
        assert m.format_version == CURRENT_FORMAT_VERSION
