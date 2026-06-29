"""Tests for LLM provider abstraction — pluggable backends."""

import json
import os
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from memory_vault.core.llm import (
    FAST_MODEL,
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
            OPENAI_API_KEY="",
        ):
            provider = get_provider()
            assert provider.name == "cloudflare"

    def test_default_openai_without_cloudflare(self):
        """When only OpenAI creds are set, get_provider() returns openai."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            assert provider.name == "openai"

    def test_env_var_overrides(self):
        """MEMORY_VAULT_LLM_PROVIDER forces a specific provider."""
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test-key",
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
        ):
            provider = get_provider()
            assert provider.name == "openai"

    def test_unknown_provider_fallback(self):
        """Unknown provider name falls back to first available."""
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="nonexistent",
            CLOUDFLARE_API_TOKEN="",
            CLOUDFLARE_ACCOUNT_ID="",
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
            OPENAI_API_KEY="",
        ):
            provider = get_provider()
            # Cloudflare with no creds: .available() == False
            assert provider.name == "cloudflare"
            assert provider.available() is False

    def test_env_var_takes_precedence(self):
        """MEMORY_VAULT_LLM_PROVIDER=openai works even with Cloudflare also configured."""
        with _WithEnv(
            MEMORY_VAULT_LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test-key",
            CLOUDFLARE_API_TOKEN="test-key",
            CLOUDFLARE_ACCOUNT_ID="test-account",
        ):
            provider = get_provider()
            assert provider.name == "openai"

    def test_clouflare_preferred_over_openai(self):
        """When both are available, Cloudflare wins (no env override)."""
        with _WithEnv(
            CLOUDFLARE_API_TOKEN="test-key",
            CLOUDFLARE_ACCOUNT_ID="test-account",
            OPENAI_API_KEY="sk-test-key",
        ):
            provider = get_provider()
            assert provider.name == "cloudflare"
