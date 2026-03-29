"""
YAML configuration validator for config.yaml.

Validates structure, required fields, and common misconfigurations.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def validate_config(cfg: dict[str, Any], config_path: str = "config.yaml") -> None:
    """
    Comprehensive validation of config.yaml structure and content.
    
    Raises ConfigValidationError if validation fails.
    Logs detailed error messages before raising.
    
    Args:
        cfg: The loaded config dictionary
        config_path: Path to config file (for error messages)
    
    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []
    warnings = []
    
    # ── Check root structure ────────────────────────────────────────────────
    if not isinstance(cfg, dict):
        errors.append(f"Config root must be a mapping, got {type(cfg).__name__}")
    
    # ── Check required top-level keys ───────────────────────────────────────
    # Support both new grouped structure (llm.providers, llm.models) and legacy flat keys
    providers = cfg.get("providers") or cfg.get("llm", {}).get("providers")
    models = cfg.get("models") or cfg.get("llm", {}).get("models")
    
    if not providers:
        errors.append(f"Missing required key: 'providers' (or 'llm.providers')")
    if not models:
        errors.append(f"Missing required key: 'models' (or 'llm.models')")
    
    # ── Validate providers section ──────────────────────────────────────────
    if providers:
        if not isinstance(providers, dict):
            errors.append(f"'providers' must be a mapping, got {type(providers).__name__}")
        else:
            for provider_name, provider_config in providers.items():
                if not isinstance(provider_config, dict):
                    errors.append(
                        f"Provider '{provider_name}' config must be a mapping, "
                        f"got {type(provider_config).__name__}"
                    )
                elif "base_url" not in provider_config:
                    errors.append(f"Provider '{provider_name}' missing required 'base_url'")
    
    # ── Validate models section ────────────────────────────────────────────
    if models:
        if not isinstance(models, dict):
            errors.append(f"'models' must be a mapping, got {type(models).__name__}")
        elif not models:
            errors.append("'models' section is empty (must define at least one model)")
        else:
            for model_name, model_config in models.items():
                if not isinstance(model_name, str):
                    errors.append(f"Model name must be a string, got {type(model_name).__name__}")
                
                if model_config is None:
                    errors.append(f"Model '{model_name}' config is empty/null")
                elif not isinstance(model_config, dict):
                    errors.append(
                        f"Model '{model_name}' config must be a mapping, "
                        f"got {type(model_config).__name__}"
                    )
                else:
                    # Validate model-level tools
                    if "tools" in model_config:
                        tools = model_config["tools"]
                        if not isinstance(tools, list):
                            errors.append(
                                f"Model '{model_name}' 'tools' must be a list, "
                                f"got {type(tools).__name__}"
                            )
                        else:
                            valid_tools = {"web_search", "visuals_core", "get_market_prices", "get_weather"}
                            for tool in tools:
                                if tool not in valid_tools:
                                    warnings.append(
                                        f"Model '{model_name}' has unknown tool '{tool}'. "
                                        f"Valid tools: {', '.join(sorted(valid_tools))}"
                                    )
                    
                    # Validate supports_tools flag
                    if "supports_tools" in model_config:
                        supports_tools = model_config["supports_tools"]
                        if not isinstance(supports_tools, bool):
                            errors.append(
                                f"Model '{model_name}' 'supports_tools' must be boolean, "
                                f"got {type(supports_tools).__name__}"
                            )
                    
                    # Validate think flag
                    if "think" in model_config:
                        think = model_config["think"]
                        if not isinstance(think, bool):
                            errors.append(
                                f"Model '{model_name}' 'think' must be boolean, "
                                f"got {type(think).__name__}"
                            )
    
    # ── Validate fallback_models ───────────────────────────────────────────
    # Support both new grouped structure (llm.fallback_models) and legacy flat keys
    fallback = cfg.get("fallback_models") or cfg.get("llm", {}).get("fallback_models")
    if fallback:
        if not isinstance(fallback, list):
            errors.append(
                f"'fallback_models' must be a list, got {type(fallback).__name__}. "
                f"Use: fallback_models:\n  - \"model1\"\n  - \"model2\""
            )
        else:
            for i, model_name in enumerate(fallback):
                if not isinstance(model_name, str):
                    errors.append(
                        f"'fallback_models[{i}]' must be a string, "
                        f"got {type(model_name).__name__}"
                    )
    
    # ── Validate scheduled_tasks section ───────────────────────────────────
    if "scheduled_tasks" in cfg:
        tasks = cfg["scheduled_tasks"]
        if not isinstance(tasks, dict):
            errors.append(
                f"'scheduled_tasks' must be a mapping, got {type(tasks).__name__}"
            )
        else:
            for task_name, task_config in tasks.items():
                if task_config is None:
                    errors.append(f"Task '{task_name}' config is empty/null")
                elif not isinstance(task_config, dict):
                    errors.append(
                        f"Task '{task_name}' config must be a mapping, "
                        f"got {type(task_config).__name__}"
                    )
                else:
                    # Check required task fields
                    if task_config.get("enabled"):
                        required_task_fields = ("cron", "model", "prompt")
                        for field in required_task_fields:
                            if field not in task_config:
                                errors.append(
                                    f"Enabled task '{task_name}' missing required field: '{field}'"
                                )
                        
                        # Check channel_id or user_id
                        if "channel_id" not in task_config and "user_id" not in task_config:
                            errors.append(
                                f"Task '{task_name}' must have either 'channel_id' or 'user_id'"
                            )
                        elif "channel_id" in task_config and "user_id" in task_config:
                            errors.append(
                                f"Task '{task_name}' has both 'channel_id' and 'user_id' "
                                f"(use only one)"
                            )
                    
                    # Validate task tools
                    if "tools" in task_config:
                        tools = task_config["tools"]
                        if not isinstance(tools, list):
                            errors.append(
                                f"Task '{task_name}' 'tools' must be a list, "
                                f"got {type(tools).__name__}"
                            )
    
    # ── Validate permissions section ───────────────────────────────────────
    if "permissions" in cfg:
        perms = cfg["permissions"]
        if not isinstance(perms, dict):
            errors.append(
                f"'permissions' must be a mapping, got {type(perms).__name__}"
            )
        else:
            for perm_type in ("users", "roles", "channels"):
                if perm_type in perms:
                    perm_config = perms[perm_type]
                    if not isinstance(perm_config, dict):
                        errors.append(
                            f"'permissions.{perm_type}' must be a mapping, "
                            f"got {type(perm_config).__name__}"
                        )
                    else:
                        for id_type in ("allowed_ids", "blocked_ids", "admin_ids"):
                            if id_type in perm_config:
                                ids = perm_config[id_type]
                                if not isinstance(ids, list):
                                    errors.append(
                                        f"'permissions.{perm_type}.{id_type}' must be a list, "
                                        f"got {type(ids).__name__}"
                                    )
    
    # ── Validate voice section (optional) ───────────────────────────────────
    # Support both new 'voice' section and legacy 'azure-speech'
    voice_cfg = cfg.get("voice") or cfg.get("azure-speech")
    voice_key = "voice" if "voice" in cfg else "azure-speech"
    
    if voice_cfg is None:
        pass  # Voice is optional
    elif not isinstance(voice_cfg, dict):
        errors.append(
            f"'{voice_key}' must be a mapping, got {type(voice_cfg).__name__}"
        )
    else:
        # Required fields
        if not voice_cfg.get("key"):
            warnings.append(f"'{voice_key}.key' is empty (TTS/STT features disabled)")
        if not voice_cfg.get("region"):
            errors.append(f"'{voice_key}.region' is required when {voice_key} is configured")
        else:
            # Optional fields validation (only if region is set)
            if "default_voice" in voice_cfg and voice_cfg["default_voice"]:
                if not isinstance(voice_cfg["default_voice"], str):
                    errors.append(
                        f"'{voice_key}.default_voice' must be a string, "
                        f"got {type(voice_cfg['default_voice']).__name__}"
                    )
            
            if "default_style" in voice_cfg and voice_cfg["default_style"]:
                valid_styles = {"cheerful", "sad", "angry", "neutral", "excited", "friendly", "terrified", "shouting", "whispering", "hopeful"}
                style = voice_cfg["default_style"]
                if style not in valid_styles:
                    warnings.append(
                        f"'{voice_key}.default_style' has unknown value '{style}'. "
                        f"Valid styles: {', '.join(sorted(valid_styles))}"
                    )
    
    # ── Log warnings ────────────────────────────────────────────────────────
    for warning in warnings:
        logger.warning("Config warning: %s", warning)
    
    # ── Log errors and exit if any ──────────────────────────────────────────
    if errors:
        error_details = "; ".join(errors)
        logger.error("=" * 70)
        logger.error("CONFIG VALIDATION FAILED (%s)", config_path)
        logger.error("=" * 70)
        for i, error in enumerate(errors, 1):
            logger.error("[%d] %s", i, error)
        logger.error("=" * 70)
        logger.error("Please fix the errors above and restart the bot.")
        logger.error("=" * 70)
        raise ConfigValidationError(f"Config validation failed: {error_details}")
