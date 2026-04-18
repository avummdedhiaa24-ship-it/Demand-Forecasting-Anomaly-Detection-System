"""
Configuration management module.
Loads and validates system configuration from YAML + environment variables.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Project root
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR:-default} patterns in config values."""
    if isinstance(value, str):
        pattern = r"\$\{([^}:]+)(?::-(.*?))?\}"
        def replacer(match: re.Match) -> str:
            env_var = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(env_var, default)
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class Config:
    """
    Singleton configuration class.
    Provides dot-notation access to nested config values.
    """

    def __init__(self, config_dict: dict[str, Any]) -> None:
        self._data = config_dict

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        val = self._data.get(key)
        if isinstance(val, dict):
            return Config(val)
        return val

    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Config(val)
        return val

    def to_dict(self) -> dict[str, Any]:
        """Return underlying dict."""
        return self._data


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Optional path to config file. Defaults to config/config.yaml.

    Returns:
        Config object with dot-notation access.
    """
    path = Path(config_path) if config_path else CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    resolved = _resolve_env_vars(raw)
    return Config(resolved)


# Module-level singleton
cfg = load_config()
