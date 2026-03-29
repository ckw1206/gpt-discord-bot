from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

import yaml

from .validator import validate_config, ConfigValidationError


DEFAULT_CONFIG_FILE = "config.yaml"
CONFIG_ENV_VAR = "CONFIG_PATH"

# Pattern to match ${ENV_VAR} in config values
ENV_VAR_PATTERN = re.compile(r'\$\{(\w+)\}')

# Track deprecated keys that have already warned (to avoid duplicate warnings)
_warned_deprecated_keys: set[str] = set()

# Mapping of legacy flat keys to new grouped structure
LEGACY_KEY_MAPPING = {
    # discord section
    "bot_token": ("discord", "bot_token"),
    "client_id": ("discord", "client_id"),
    "status_message": ("discord", "status_message"),
    "permissions": ("discord", "permissions"),
    # behavior section
    "max_text": ("behavior", "max_text"),
    "max_images": ("behavior", "max_images"),
    "max_messages": ("behavior", "max_messages"),
    "use_plain_responses": ("behavior", "use_plain_responses"),
    "show_embed_color": ("behavior", "show_embed_color"),
    "allow_dms": ("behavior", "allow_dms"),
    # llm section
    "providers": ("llm", "providers"),
    "models": ("llm", "models"),
    "fallback_models": ("llm", "fallback_models"),
    "persona": ("llm", "persona"),
    "system_prompt": ("llm", "system_prompt"),
    # voice section (rename)
    "azure-speech": ("voice", None),  # None means move entire key to new section
}


def _warn_deprecated_key(legacy_key: str, new_path: str) -> None:
    """
    Log a deprecation warning for a legacy config key.
    Only warns once per key to avoid duplicate warnings.
    """
    global _warned_deprecated_keys
    if legacy_key in _warned_deprecated_keys:
        return
    _warned_deprecated_keys.add(legacy_key)
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Config key '{legacy_key}' is deprecated. "
        f"Use '{new_path}' instead. This key will be removed in a future version."
    )


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize config from legacy flat structure to new grouped structure.
    
    This converts old keys like 'bot_token' to new keys like 'discord.bot_token'.
    Both old and new formats are supported - this normalizes everything to the
    new grouped structure for consistent access throughout the codebase.
    
    If a key exists in both old and new locations, the new location takes precedence.
    """
    result = dict(cfg)  # Make a copy to avoid mutating original
    
    for legacy_key, (new_section, new_key) in LEGACY_KEY_MAPPING.items():
        # Skip if legacy key doesn't exist
        if legacy_key not in result:
            continue
            
        # Get the value from legacy key
        value = result.pop(legacy_key)
        
        # Determine the target path
        if new_key is None:
            # Move entire key to new section (e.g., azure-speech -> voice)
            target_path = new_section
        else:
            target_path = f"{new_section}.{new_key}"
        
        # Check if new location already exists (new format takes precedence)
        if new_section in result and new_key and new_key in result[new_section]:
            # New format exists, log deprecation for legacy but use new
            _warn_deprecated_key(legacy_key, target_path)
            # Remove the legacy value since we're using new format
            continue
        
        # Handle azure-speech -> voice specially (move entire dict)
        if new_key is None:
            if new_section not in result:
                result[new_section] = value
            else:
                # Merge if both exist
                if isinstance(value, dict) and isinstance(result[new_section], dict):
                    result[new_section].update(value)
                else:
                    result[new_section] = value
            _warn_deprecated_key(legacy_key, target_path)
        else:
            # Create nested structure
            if new_section not in result:
                result[new_section] = {}
            result[new_section][new_key] = value
            _warn_deprecated_key(legacy_key, target_path)
    
    # Add backward compatibility aliases for top-level keys that code expects
    # This allows code using config["models"] and config["providers"] to work
    if "llm" in result:
        if "models" not in result and "models" in result.get("llm", {}):
            result["models"] = result["llm"]["models"]
        if "providers" not in result and "providers" in result.get("llm", {}):
            result["providers"] = result["llm"]["providers"]
    
    # Add backward compatibility aliases for discord keys
    if "discord" in result:
        if "bot_token" not in result and "bot_token" in result.get("discord", {}):
            result["bot_token"] = result["discord"]["bot_token"]
        if "client_id" not in result and "client_id" in result.get("discord", {}):
            result["client_id"] = result["discord"]["client_id"]
        if "status_message" not in result and "status_message" in result.get("discord", {}):
            result["status_message"] = result["discord"]["status_message"]
        if "permissions" not in result and "permissions" in result.get("discord", {}):
            result["permissions"] = result["discord"]["permissions"]
    
    # ── Type Fixes ───────────────────────────────────────────────────────
    # Auto-fix common misconfigurations that would fail validation
    
    # Fix fallback_models: convert string to list
    for path in ["llm.fallback_models", "fallback_models"]:
        parts = path.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current = None
                break
            current = current[part]
        
        if current is not None and parts[-1] in current:
            value = current[parts[-1]]
            if isinstance(value, str):
                # Convert "model1, model2" or "model1" to list
                current[parts[-1]] = [s.strip() for s in value.split(",") if s.strip()]
    
    return result


def _interpolate_env_vars(value: Any) -> Any:
    """
    Recursively replace ${VAR} patterns with environment variable values.
    
    Supports strings, lists, and dicts.
    """
    if isinstance(value, str):
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Keep original if env var not set
        return ENV_VAR_PATTERN.sub(replace_var, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def get_config_path() -> str:
    """
    Resolve the config path, preferring an explicit environment override.
    """
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return env_path
    return DEFAULT_CONFIG_FILE


def _load_raw_yaml(path: str | None = None) -> dict[str, Any]:
    """Load raw YAML config without interpolation."""
    cfg_path = path or get_config_path()
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.error("Config file not found: %s", cfg_path)
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error("YAML parsing error in %s", cfg_path, e)
        sys.exit(1)
    
    if not isinstance(data, dict):
        logging.error("Config root must be a mapping, got %s", type(data).__name__)
        sys.exit(1)
    
    return data


def get_config(path: str | None = None) -> dict[str, Any]:
    """
    Public helper for loading configuration.

    - Respects CONFIG_PATH if set.
    - Performs comprehensive YAML validation.
    - Exits with error code 1 if validation fails.
    - Returns the raw dict with ${VAR} interpolated from environment variables.
    - Normalizes legacy flat keys to new grouped structure.
    """
    cfg_path = path or get_config_path()
    cfg = _load_raw_yaml(cfg_path)
    cfg = _interpolate_env_vars(cfg)
    cfg = _normalize_config(cfg)  # Normalize legacy keys to new grouped structure
    
    try:
        validate_config(cfg, cfg_path)
    except ConfigValidationError:
        sys.exit(1)
    
    return cfg

