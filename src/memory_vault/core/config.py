"""Simple JSON-based config persistence for memory-vault."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "memory-vault"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def read_config() -> dict[str, Any]:
    """Read config file, return dict (empty if missing)."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def write_config(config: dict[str, Any]) -> None:
    """Write config dict to JSON file."""
    _ensure_dir()
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a config value by dot-separated key (e.g. 'llm.provider')."""
    config = read_config()
    parts = key.split(".")
    val: Any = config
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return default
    return val if val is not None else default


def set_config_value(key: str, value: Any) -> None:
    """Set a config value by dot-separated key, persisting to JSON."""
    config = read_config()
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    write_config(config)


def get_llm_provider() -> str | None:
    """Resolve the LLM provider name from env (highest priority) or config."""
    env_provider = os.environ.get("MEMORY_VAULT_LLM_PROVIDER", "").strip()
    if env_provider:
        return env_provider
    return get_config_value("llm.provider", None)
