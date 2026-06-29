"""Tests for LLM provider abstraction — pluggable backends."""

import json
import os
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from memory_vault.core.llm import (
    FAST_MODEL,
    AnthropicProvider,
    CloudflareAI,
    LLMProvider,
    OpenAICompatibleProvider,
    get_provider,
)

# ── Helpers ─────────────────────────────────────────────────────────


class _WithEnv:
    """Context manager to temporarily set/clear env vars."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._old = {}

    def __enter__(self):
        for k, v in self.kwargs.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *args):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _mock_urlopen(response_data: dict, status: int = 200):
    """Patch urllib.request.urlopen to return a canned JSON response."""
    body = json.dumps(response_data).encode()
    cm = MagicMock()
    cm.__enter__.return_value = cm
    cm.read.return_value = body
    cm.status = status
    return cm


# ── Interface compliance ────────────────────────────────────────────


class TestLLMProviderInterface:
    """Verify the ABC enforces the right contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LLMProvider()  # abstract

    def test_minimal_subclass(self):
        class Minimal(LLMProvider):
            name = "test"

            def available(self) -> bool:
                return False

            def default_model(self) -> str:
                return "test-model"

            def chat(self, messages, model="", max_tokens=512, temperature=0.3, timeout=120):
                return None

        inst = Minimal()
        assert inst.name == "test"
        assert inst.available() is False
        assert inst.default_model() == "test-model"
        assert inst.chat([], model="x") is None


# ── Cloudflare provider ─────────────────────────────────────────────


class TestCloudflareAI:
    def test_available_with_creds(self):
        with _WithEnv(CLOUDFLARE_API_TOKEN="test-key", CLOUDFLARE_ACCOUNT_ID="test-account"):
            cf = CloudflareAI()
            assert cf.available() is True

    def test_unavailable_without_key(self):
        with _WithEnv(CLOUDFLARE_API_TOKEN="", CLOUDFLARE_ACCOUNT_ID="test-account"):
            cf = CloudflareAI()
            assert cf.available() is False

    def test_unavailable_without_account(self):
        with _WithEnv(CLOUDFLARE_API_TOKEN="test-key", CLOUDFLARE_ACCOUNT_ID=""):
            cf = CloudflareAI()
            assert cf.available() is False

    def test_unavailable_absent_vars(self):
        """available() is False when env vars are not set at all."""
        with _WithEnv(CLOUDFLARE_API_TOKEN=None, CLOUDFLARE_ACCOUNT_ID=None):
            cf = CloudflareAI()
            assert cf.available() is False

    def test_default_model(self):
        cf = CloudflareAI()
        assert cf.default_model() == FAST_MODEL

    def test_name(self):
        cf = CloudflareAI()
        assert cf.name == "cloudflare"

    def test_chat_calls_api(self):
        with _WithEnv(CLOUDFLARE_API_TOKEN="test-key", CLOUDFLARE_ACCOUNT_ID="test-account"):
            cf = CloudflareAI()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({
                    "success": True,
                    "result": {"response": "Hello from mock!"},
                })
                mock_urlopen.return_value = mock_response

                result = cf.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model=FAST_MODEL,
                )
                assert result == "Hello from mock!"

    def test_chat_returns_none_on_unsuccessful(self):
        """API returns success=False → returns None."""
        with _WithEnv(CLOUDFLARE_API_TOKEN="test-key", CLOUDFLARE_ACCOUNT_ID="test-account"):
            cf = CloudflareAI()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({
                    "success": False,
                    "errors": [{"message": "rate limited"}],
                })
                mock_urlopen.return_value = mock_response
                result = cf.chat(messages=[{"role": "user", "content": "hi"}], model=FAST_MODEL)
                assert result is None

    def test_chat_returns_none_on_error(self):
        with _WithEnv(CLOUDFLARE_API_TOKEN="test-key", CLOUDFLARE_ACCOUNT_ID="test-account"):
            cf = CloudflareAI()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = URLError("connection error")
                result = cf.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model=FAST_MODEL,
                )
                assert result is None

    def test_chat_no_key_graceful(self):
        """Without credentials, chat() returns None, not crash."""
        with _WithEnv(CLOUDFLARE_API_TOKEN="", CLOUDFLARE_ACCOUNT_ID=""):
            cf = CloudflareAI()
            result = cf.chat(messages=[{"role": "user", "content": "hi"}], model=FAST_MODEL)
            assert result is None


# ── Anthropic provider ──────────────────────────────────────────────


class TestAnthropicProvider:
    def test_available_with_key(self):
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key"):
            provider = AnthropicProvider()
            assert provider.available() is True

    def test_unavailable_without_key(self):
        with _WithEnv(ANTHROPIC_API_KEY=""):
            provider = AnthropicProvider()
            assert provider.available() is False

    def test_unavailable_absent_key(self):
        with _WithEnv(ANTHROPIC_API_KEY=None):
            provider = AnthropicProvider()
            assert provider.available() is False

    def test_default_model(self):
        """Default is claude-sonnet-4 when ANTHROPIC_MODEL is not set."""
        with _WithEnv(ANTHROPIC_MODEL=None):
            provider = AnthropicProvider()
            assert provider.default_model() == "claude-sonnet-4"

    def test_default_model_from_env(self):
        with _WithEnv(ANTHROPIC_MODEL="claude-haiku-3.5"):
            provider = AnthropicProvider()
            assert provider.default_model() == "claude-haiku-3.5"

    def test_name(self):
        provider = AnthropicProvider()
        assert provider.name == "anthropic"

    def test_custom_base_url(self):
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key", ANTHROPIC_BASE_URL="https://myproxy.anthropic.com/v1"):
            provider = AnthropicProvider()
            assert provider.available() is True

    def test_chat_calls_api(self):
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key"):
            provider = AnthropicProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({
                    "content": [{"type": "text", "text": "Hello from Claude!"}]
                })
                mock_urlopen.return_value = mock_response

                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4",
                )
                assert result == "Hello from Claude!"

    def test_chat_system_message_conversion(self):
        """System messages are extracted to the separate Anthropic 'system' field."""
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key"):
            provider = AnthropicProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({
                    "content": [{"type": "text", "text": "Roger!"}]
                })
                mock_urlopen.return_value = mock_response

                result = provider.chat(
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hi"},
                    ],
                    model="claude-sonnet-4",
                )
                assert result == "Roger!"
                # Verify system field was sent separately
                call_body = json.loads(mock_urlopen.call_args[0][0].data)
                assert call_body.get("system") == "You are helpful."
                # Verify system role is not in messages
                roles = [m["role"] for m in call_body["messages"]]
                assert "system" not in roles
                assert roles == ["user"]

    def test_chat_assistant_first_message(self):
        """If conversation starts with assistant, a user placeholder is prepended."""
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key"):
            provider = AnthropicProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({
                    "content": [{"type": "text", "text": "continuing..."}]
                })
                mock_urlopen.return_value = mock_response
                provider.chat(
                    messages=[{"role": "assistant", "content": "Hello!"}],
                    model="claude-sonnet-4",
                )
                call_body = json.loads(mock_urlopen.call_args[0][0].data)
                assert call_body["messages"][0]["role"] == "user"

    def test_chat_empty_content(self):
        """API returns content: [] → returns None."""
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key"):
            provider = AnthropicProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({"content": []})
                mock_urlopen.return_value = mock_response
                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4",
                )
                assert result is None

    def test_chat_returns_none_on_error(self):
        with _WithEnv(ANTHROPIC_API_KEY="sk-ant-test-key"):
            provider = AnthropicProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = URLError("connection error")
                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="claude-sonnet-4",
                )
                assert result is None

    def test_chat_no_key_graceful(self):
        with _WithEnv(ANTHROPIC_API_KEY=""):
            provider = AnthropicProvider()
            result = provider.chat(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4",
            )
            assert result is None


# ── OpenAI-compatible provider ──────────────────────────────────────


class TestOpenAICompatibleProvider:
    def test_available_with_key(self):
        with _WithEnv(OPENAI_API_KEY="sk-test-key"):
            provider = OpenAICompatibleProvider()
            assert provider.available() is True

    def test_unavailable_without_key(self):
        with _WithEnv(OPENAI_API_KEY=""):
            provider = OpenAICompatibleProvider()
            assert provider.available() is False

    def test_unavailable_absent_key(self):
        with _WithEnv(OPENAI_API_KEY=None):
            provider = OpenAICompatibleProvider()
            assert provider.available() is False

    def test_default_model(self):
        """Default is gpt-4o-mini when OPENAI_MODEL is not set."""
        with _WithEnv(OPENAI_MODEL=None):
            provider = OpenAICompatibleProvider()
            assert provider.default_model() == "gpt-4o-mini"

    def test_default_model_from_env(self):
        with _WithEnv(OPENAI_MODEL="gpt-4o"):
            provider = OpenAICompatibleProvider()
            assert provider.default_model() == "gpt-4o"

    def test_name(self):
        provider = OpenAICompatibleProvider()
        assert provider.name == "openai"

    def test_custom_base_url(self):
        with _WithEnv(OPENAI_API_KEY="sk-test-key", OPENAI_BASE_URL="https://myproxy.com/v1"):
            provider = OpenAICompatibleProvider()
            assert provider.available() is True

    def test_chat_calls_api(self):
        with _WithEnv(OPENAI_API_KEY="sk-test-key"):
            provider = OpenAICompatibleProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({
                    "choices": [{"message": {"content": "Hello from OpenAI!"}}]
                })
                mock_urlopen.return_value = mock_response

                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o-mini",
                )
                assert result == "Hello from OpenAI!"

    def test_chat_empty_choices(self):
        """API returns empty choices → returns None."""
        with _WithEnv(OPENAI_API_KEY="sk-test-key"):
            provider = OpenAICompatibleProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_response = _mock_urlopen({"choices": []})
                mock_urlopen.return_value = mock_response
                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o-mini",
                )
                assert result is None

    def test_chat_returns_none_on_error(self):
        with _WithEnv(OPENAI_API_KEY="sk-test-key"):
            provider = OpenAICompatibleProvider()
            with patch("memory_vault.core.llm.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = URLError("connection error")
                result = provider.chat(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o-mini",
                )
                assert result is None

    def test_chat_no_key_graceful(self):
        with _WithEnv(OPENAI_API_KEY=""):
            provider = OpenAICompatibleProvider()
            result = provider.chat(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
            )
            assert result is None


# ── Provider registry ───────────────────────────────────────────────


class TestGetProvider:
    def test_default_cloudflare_with_creds(self):
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="test-key",
            CLOUDFLARE_ACCOUNT_ID="test-account",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="",
        ):
            provider = get_provider()
            assert provider.name == "cloudflare"

    def test_default_anthropic_without_cloudflare(self):
        """When only Anthropic creds are set, get_provider() returns anthropic."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            ANTHROPIC_API_KEY="sk-ant-test-key",
            OPENAI_API_KEY="",
        ):
            provider = get_provider()
            assert provider.name == "anthropic"

    def test_default_openai_without_others(self):
        """When only OpenAI creds are set, get_provider() returns openai."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            assert provider.name == "openai"

    def test_anthropic_preferred_over_openai(self):
        """Anthropic wins over OpenAI when both are available (Cloudflare not)."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            ANTHROPIC_API_KEY="sk-ant-test-key",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            assert provider.name == "anthropic"

    def test_cloudflare_still_wins(self):
        """Cloudflare still wins over Anthropic when both are available."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="test-key",
            CLOUDFLARE_ACCOUNT_ID="test-account",
            ANTHROPIC_API_KEY="sk-ant-test-key",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            assert provider.name == "cloudflare"

    def test_env_var_overrides(self):
        """MEMORY_VAULT_LLM_PROVIDER forces a specific provider."""
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test-key",
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            ANTHROPIC_API_KEY="",
        ):
            provider = get_provider()
            assert provider.name == "openai"
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="anthropic",
            ANTHROPIC_API_KEY="sk-ant-test-key",
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            OPENAI_API_KEY="",
        ):
            provider = get_provider()
            assert provider.name == "anthropic"

    def test_unknown_provider_fallback(self):
        """Unknown provider name falls back to first available."""
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="nonexistent",
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            # The env name is invalid, so auto-detect runs → OpenAI is available
            assert provider.name == "openai"

    def test_no_provider_available(self):
        """Without any creds, returns the none-provider."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            ANTHROPIC_API_KEY="",
            OPENAI_API_KEY="",
        ):
            provider = get_provider()
            # Cloudflare with no creds: .available() == False
            assert provider.name == "cloudflare"
            assert provider.available() is False

    def test_env_var_takes_precedence(self):
        """MEMORY_VAULT_LLM_PROVIDER works even when others are configured."""
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="anthropic",
            ANTHROPIC_API_KEY="sk-ant-test-key",
            CLOUDFLARE_API_TOKEN="test-key",
            CLOUDFLARE_ACCOUNT_ID="test-account",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            assert provider.name == "anthropic"
