"""LLM Provider abstraction — pluggable backends for narration & indexing.

Memory Vault uses LLMs for two jobs:
  1. Session narration (compression → narrative + decisions + handoff)
  2. Session indexing (title + summary generation)

Both consume the same provider interface, so swapping the backend
is a config change, not a code change.

Providers available:
  - cloudflare  (default) — Cloudflare Workers AI
  - anthropic   — Anthropic Claude (via x-api-key auth)
  - openai      — any OpenAI-compatible API (OpenAI, OpenRouter, etc.)

Selection (priority):
  1. MEMORY_VAULT_LLM_PROVIDER env var (e.g. "anthropic")
  2. CLOUDFLARE credentials → cloudflare
  3. ANTHROPIC_API_KEY → anthropic
  4. OpenAI-compatible credentials → openai
  5. None → template fallback (always works)
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from urllib.error import URLError
from urllib.request import Request, urlopen

if sys.version_info >= (3, 12):
    from typing import override
else:
    def override(fn):
        return fn


# ── Models ─────────────────────────────────────────────────────────

FAST_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
"""Default fast model for summaries & indexing."""

DEEP_MODEL = "@cf/zai-org/glm-5.2"
"""Slower but richer reasoning model for narration."""

_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4"


# ── Base Provider ───────────────────────────────────────────────────


class LLMProvider(ABC):
    """Pluggable LLM backend — implement this to add a new provider.

    Usage in SessionNarrator / SessionIndex::

        provider = get_provider()
        result = provider.chat(messages, model="...")
    """

    name: str = "abstract"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str | None:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format message list.
            model: Model identifier (provider-specific).
            max_tokens: Max tokens in the response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.

        Returns:
            The response text, or None on failure.
        """
        ...

    @abstractmethod
    def available(self) -> bool:
        """Check if this provider's credentials are configured."""
        ...

    @abstractmethod
    def default_model(self) -> str:
        """The sensible default model for this provider."""
        ...


# ── Cloudflare Workers AI ───────────────────────────────────────────


class CloudflareAI(LLMProvider):
    """Cloudflare Workers AI — free tier, fast inference.

    Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in env.
    """

    name = "cloudflare"

    # ── Lazy credential resolution ──────────────────────────────
    # Credential checks happen at call time (available(), chat())
    # not at construction, so env-var changes by tests or the
    # user's runtime environment are always reflected.

    @override
    def available(self) -> bool:
        return bool(
            os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            and os.environ.get("CLOUDFLARE_API_TOKEN", "")
        )

    @override
    def default_model(self) -> str:
        return FAST_MODEL

    @override
    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str | None:
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not (account_id and api_token):
            return None

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{model}"
        )
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        req = Request(url, data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {api_token}")
        req.add_header("Content-Type", "application/json")

        try:
            resp = urlopen(req, timeout=timeout)
            body = json.loads(resp.read().decode())
            if body.get("success"):
                return body["result"].get("response", "")
            return None
        except (URLError, json.JSONDecodeError, OSError):
            return None


# ── Anthropic Provider ──────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API.

    Requires ANTHROPIC_API_KEY in env.
    ANTHROPIC_BASE_URL and ANTHROPIC_MODEL are optional overrides.

    Accepts OpenAI-format messages internally (including ``role: "system"``)
    and transforms them to the Anthropic Messages API format.
    """

    name = "anthropic"

    def __init__(self):
        self.base_url = (
            os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            .rstrip("/")
        )

    @override
    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY", ""))

    @override
    def default_model(self) -> str:
        return os.environ.get("ANTHROPIC_MODEL", _ANTHROPIC_DEFAULT_MODEL)

    @override
    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        # ── Convert OpenAI-format messages to Anthropic format ──────
        system, converted = self._convert_messages(messages)

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": converted,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature

        url = f"{self.base_url}/messages"
        data = json.dumps(payload).encode()

        req = Request(url, data=data, method="POST")
        req.add_header("x-api-key", api_key)
        req.add_header("Content-Type", "application/json")
        req.add_header("anthropic-version", "2023-06-01")

        try:
            resp = urlopen(req, timeout=timeout)
            body = json.loads(resp.read().decode())
            content = body.get("content", [])
            if content:
                return content[0].get("text", "")
            return None
        except (URLError, json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _convert_messages(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-format messages to Anthropic format.

        Returns:
            (system_prompt_or_none, anthropic_messages)
        """
        system_parts: list[str] = []
        converted: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system messages for the separate "system" field
            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            # Map roles: "tool" → "user" (Anthropic has no tool role in basic usage)
            anth_role = role if role in ("user", "assistant") else "user"

            converted.append({"role": anth_role, "content": content})

        # Anthropic requires the first message to be from "user"
        if converted and converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": "(continue)"})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, converted


# ── OpenAI-Compatible Provider ──────────────────────────────────────


class OpenAICompatibleProvider(LLMProvider):
    """Any OpenAI-compatible API: OpenAI, OpenRouter, Anthropic, etc.

    Reads from env:
      OPENAI_BASE_URL  — base URL (default: https://api.openai.com/v1)
      OPENAI_API_KEY   — API key
      OPENAI_MODEL     — default model (default: gpt-4o-mini)

    To use OpenRouter::
        OPENAI_BASE_URL=https://openrouter.ai/api/v1
        OPENAI_API_KEY=sk-or-...
        OPENAI_MODEL=anthropic/claude-sonnet-4
    """

    name = "openai"

    def __init__(self):
        self.base_url = (
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            .rstrip("/")
        )

    @override
    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY", ""))

    @override
    def default_model(self) -> str:
        return os.environ.get("OPENAI_MODEL", _OPENAI_DEFAULT_MODEL)

    @override
    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str | None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None

        url = f"{self.base_url}/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        req = Request(url, data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")

        try:
            resp = urlopen(req, timeout=timeout)
            body = json.loads(resp.read().decode())
            choices = body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
        except (URLError, json.JSONDecodeError, OSError):
            return None


# ── Provider registry ───────────────────────────────────────────────

_PROVIDERS: dict[str, LLMProvider] = {}
"""Registered provider instances, keyed by name."""


def register_provider(provider: LLMProvider) -> None:
    """Register an LLM provider instance."""
    _PROVIDERS[provider.name] = provider


def get_provider(name: str | None = None) -> LLMProvider:
    """Get a provider by name, or auto-detect the best available.

    Auto-detection priority:
      1. ``MEMORY_VAULT_LLM_PROVIDER`` env var
      2. ``llm.provider`` in ``~/.config/memory-vault/config.yaml``
      3. Cloudflare (if credentials present)
      4. Anthropic (if API key present)
      5. OpenAI-compatible (if API key present)
      6. Cloudflare (no-cred fallback — .available() = False)
    """
    # Named lookup
    if name:
        return _PROVIDERS[name]

    # Env override
    env_name = os.environ.get("MEMORY_VAULT_LLM_PROVIDER", "").strip().lower()
    if env_name and env_name in _PROVIDERS:
        return _PROVIDERS[env_name]

    # Config override (only if the configured provider is actually available)
    try:
        from memory_vault.core.config import get_llm_provider
        cfg_name = get_llm_provider()
        if cfg_name and cfg_name.lower() in _PROVIDERS:
            candidate = _PROVIDERS[cfg_name.lower()]
            if candidate.available():
                return candidate
    except Exception:
        pass

    # Auto-detect by availability
    for p in ("cloudflare", "anthropic", "openai"):
        candidate = _PROVIDERS.get(p)
        if candidate and candidate.available():
            return candidate

    # Fallback: Cloudflare (allows template fallback downstream)
    return _PROVIDERS.get("cloudflare", CloudflareAI())


def list_providers() -> dict[str, bool]:
    """Return {name: available} for all registered providers."""
    return {n: p.available() for n, p in _PROVIDERS.items()}


# ── Auto-register built-in providers ───────────────────────────────

register_provider(CloudflareAI())
register_provider(AnthropicProvider())
register_provider(OpenAICompatibleProvider())
