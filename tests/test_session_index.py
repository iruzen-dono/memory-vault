"""Tests for SessionIndex — local SQLite cache of enriched session titles.

LLM calls are mocked to keep tests hermetic and fast.
"""

from pathlib import Path

from memory_vault.core.llm import LLMProvider
from memory_vault.core.session_index import SessionIndex

# ── Helper: mock the provider ───────────────────────────────────────

class _MockProvider(LLMProvider):
    """A fake provider that is never available — forces template fallback."""
    name = "mock"

    def available(self) -> bool:
        return False

    def default_model(self) -> str:
        return "mock-model"

    def chat(self, messages, model="", max_tokens=512, temperature=0.3, timeout=120):
        return None


def _no_provider(idx: SessionIndex) -> SessionIndex:
    """Replace the provider with an unavailable mock."""
    idx._provider = _MockProvider()
    return idx


def _make_idx(tmp_path: Path, name: str = "test.db") -> SessionIndex:
    idx = SessionIndex(tmp_path / name)
    _no_provider(idx)
    return idx


# ── Core index operations ───────────────────────────────────────────


class TestSessionIndex:
    def test_init_creates_db(self, tmp_path: Path) -> None:
        db = tmp_path / "test-index.db"
        idx = _make_idx(tmp_path, "test-index.db")
        assert db.exists(), "DB file should be created"
        assert idx.count() == 0
        assert idx.summary_stats() == {"total": 0, "with_summary": 0}

    def test_upsert_and_get(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        idx._upsert("ses1", "My Title", "A short summary.", "test-model", 42)
        entry = idx.get("ses1")
        assert entry is not None
        assert entry["title"] == "My Title"
        assert entry["summary"] == "A short summary."
        assert entry["model_used"] == "test-model"
        assert entry["msg_count"] == 42
        assert entry["indexed_at"] > 0

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        assert idx.get("nonexistent") is None

    def test_upsert_overwrites(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        idx._upsert("s1", "Old", "Old summary", "m1", 10)
        idx._upsert("s1", "New", "New summary", "m2", 20)
        entry = idx.get("s1")
        assert entry["title"] == "New"
        assert entry["summary"] == "New summary"
        assert entry["msg_count"] == 20

    def test_count(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        assert idx.count() == 0
        idx._upsert("a", "A", "", "m", 1)
        assert idx.count() == 1
        idx._upsert("b", "B", "", "m", 1)
        assert idx.count() == 2

    def test_summary_stats(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        assert idx.summary_stats() == {"total": 0, "with_summary": 0}
        idx._upsert("a", "A", "", "m", 1)
        idx._upsert("b", "B", "has text", "m", 1)
        stats = idx.summary_stats()
        assert stats["total"] == 2
        assert stats["with_summary"] == 1

    def test_remove(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        idx._upsert("x", "X", "", "m", 1)
        assert idx.count() == 1
        idx.remove("x")
        assert idx.count() == 0
        assert idx.get("x") is None

    def test_clear(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        idx._upsert("a", "A", "", "m", 1)
        idx._upsert("b", "B", "", "m", 1)
        assert idx.count() == 2
        idx.clear()
        assert idx.count() == 0

    def test_list_indexed_order(self, tmp_path: Path) -> None:
        import time
        idx = _make_idx(tmp_path)
        idx._upsert("old", "Old", "", "m", 1)
        time.sleep(0.01)
        idx._upsert("new", "New", "", "m", 1)
        entries = idx.list_indexed(5)
        assert entries[0]["session_id"] == "new"
        assert entries[1]["session_id"] == "old"

    def test_set_model(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        assert idx.model != ""
        idx.set_model("custom-model")
        assert idx.model == "custom-model"


# ── LLM response parsing ────────────────────────────────────────────


class TestExtractTitleSummary:
    def test_parses_valid_response(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_parse1.db")
        response = "TITLE: Fix Login Bug\nSUMMARY: Resolved a token expiry issue in auth module."
        title, summary = idx._extract_title_summary(response)
        assert title == "Fix Login Bug"
        assert summary == "Resolved a token expiry issue in auth module."

    def test_fallback_no_markers(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_parse2.db")
        response = "This is a loose response without formatting markers"
        title, summary = idx._extract_title_summary(response)
        assert title == "This is a loose response without formatting markers"
        assert summary == ""

    def test_truncation(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_parse3.db")
        long_title = "A" * 100
        long_summary = "B" * 600
        response = f"TITLE: {long_title}\nSUMMARY: {long_summary}"
        title, summary = idx._extract_title_summary(response)
        assert len(title) <= 60
        assert len(summary) <= 500

    def test_case_insensitive(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_parse4.db")
        response = "title: lowercase title\nsummary: lowercase summary here"
        title, summary = idx._extract_title_summary(response)
        assert title == "lowercase title"
        assert summary == "lowercase summary here"

    def test_extra_whitespace(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_parse5.db")
        response = "  TITLE:  Spaced Title  \n  SUMMARY:  Spaced Summary  "
        title, summary = idx._extract_title_summary(response)
        assert title == "Spaced Title"
        assert summary == "Spaced Summary"

    def test_quotes_stripped(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_parse6.db")
        response = 'TITLE: "Quoted Title"\nSUMMARY: \'Quoted Summary\''
        title, summary = idx._extract_title_summary(response)
        assert title == "Quoted Title"
        assert summary == "Quoted Summary"


# ── Transcript building ─────────────────────────────────────────────


class TestBuildTranscript:
    def test_basic_messages(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_trans1.db")
        messages = [
            {"role": "user", "content": "Hello, can you help?"},
            {"role": "assistant", "content": "Sure!", "tool_calls": '[{"name": "web_search"}]'},
            {"role": "tool", "content": "results here", "tool_name": "web_search"},
        ]
        sample = idx._build_transcript_sample(messages)
        assert "User: Hello" in sample
        assert "Assistant: Sure!" in sample
        assert "[tool: web_search]" in sample
        assert "[result: web_search]" in sample

    def test_tool_calls_as_dict(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_trans2.db")
        messages = [
            {"role": "assistant", "content": "", "tool_calls": {"name": "read_file"}},
        ]
        sample = idx._build_transcript_sample(messages)
        assert "[tool: read_file]" in sample

    def test_empty_messages(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_trans3.db")
        assert idx._build_transcript_sample([]) == ""

    def test_large_cap(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_trans4.db")
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(200)]
        sample = idx._build_transcript_sample(messages)
        assert sample.count("User:") <= 80
        lines = sample.split("\n")
        assert len(lines) <= 100

    def test_assistant_no_tool_calls(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path, "_trans5.db")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]
        sample = idx._build_transcript_sample(messages)
        assert "Assistant: hello back" in sample
        assert "[tool:" not in sample


# ── Session indexing ────────────────────────────────────────────────


class TestIndexSession:
    def test_creates_entry_template_fallback(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        session = {"id": "s1", "title": "Raw Title", "model": "gpt4"}
        messages = [
            {"role": "user", "content": "Build a login page"},
            {"role": "assistant", "content": "Here's the code"},
        ]
        entry = idx.index_session(session, messages)
        assert entry is not None
        assert entry["session_id"] == "s1"
        assert entry["title"] == "Raw Title"
        assert entry["summary"] == "Build a login page"
        assert entry["model_used"] == "template"
        assert entry["msg_count"] == 2

    def test_skips_if_recent(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        session = {"id": "s1", "title": "T", "model": "m"}
        msg10 = [{"role": "user", "content": "m"}] * 10
        msg5 = [{"role": "user", "content": "m"}] * 5
        idx.index_session(session, msg10)
        entry = idx.index_session(session, msg5)
        assert entry["msg_count"] == 10

    def test_force_reindex(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        session = {"id": "s1", "title": "Original", "model": "m"}
        msg5 = [{"role": "user", "content": "m"}] * 5
        idx.index_session(session, msg5)
        entry = idx.index_session(session, msg5, force=True)
        assert entry is not None
        assert entry["msg_count"] == 5
        assert entry["model_used"] == "template"

    def test_empty_session_id(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        assert idx.index_session(
            {"id": "", "title": "t", "model": "m"}, []
        ) is None

    def test_summary_empty_fallback_to_tools(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        session = {"id": "s1", "title": "Tool Session", "model": "m"}
        messages = [
            {"role": "tool", "content": "result", "tool_name": "web_search"},
        ]
        entry = idx.index_session(session, messages)
        assert entry is not None
        assert "messages" in entry["summary"] or "tool" in entry["summary"]

    def test_explicit_model_passed(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        idx.set_model("my-custom-model")
        assert idx.model == "my-custom-model"

    def test_model_fallback_chain(self, tmp_path: Path, monkeypatch) -> None:
        idx = _make_idx(tmp_path)
        assert idx.model != ""

        monkeypatch.setenv("MEMORY_VAULT_INDEX_MODEL", "env-model")
        idx2 = SessionIndex(tmp_path / "test2.db")
        _no_provider(idx2)
        assert idx2.model == "env-model"

        idx2.set_model("explicit-model")
        assert idx2.model == "explicit-model"

    def test_no_messages(self, tmp_path: Path) -> None:
        idx = _make_idx(tmp_path)
        session = {"id": "s1", "title": "Empty Session", "model": "m"}
        entry = idx.index_session(session, [])
        assert entry is not None
        assert entry["msg_count"] == 0
        assert entry["title"] == "Empty Session"
