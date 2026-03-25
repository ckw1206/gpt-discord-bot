"""
Voice module configuration loader.

Parses voice section (or legacy azure-speech) from config.yaml 
and supports environment variable overrides.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger("discord-bot.voice")

# Pattern to match ${ENV_VAR} in config values
ENV_VAR_PATTERN = re.compile(r'\$\{(\w+)\}')

# Track deprecation warnings to avoid repeating
_azure_speech_warned = False


def _get_with_fallback(config_value: str, env_var: str) -> str:
    """
    Get config value with environment variable fallback.
    
    If config_value is empty, falls back to env_var environment variable.
    Also handles ${VAR} interpolation syntax.
    """
    # First apply ${VAR} interpolation if present
    if isinstance(config_value, str) and ENV_VAR_PATTERN.search(config_value):
        config_value = ENV_VAR_PATTERN.sub(
            lambda m: os.getenv(m.group(1), m.group(0)), 
            config_value
        )
    
    # If config value is empty, fall back to environment variable
    if not config_value:
        return os.getenv(env_var, "")
    
    return config_value


def _interpolate_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    if not isinstance(value, str):
        return value
    
    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    return ENV_VAR_PATTERN.sub(replace_var, value)


@dataclass
class VoiceConfig:
    """Configuration for Azure Speech Services."""
    key: str
    region: str
    endpoint: Optional[str] = None
    default_voice: Optional[str] = None
    default_style: Optional[str] = None
    
    @property
    def is_configured(self) -> bool:
        """Check if Azure Speech is properly configured."""
        return bool(self.key and self.region)


def get_voice_config(config: dict[str, Any]) -> Optional[VoiceConfig]:
    """
    Parse voice section from config (supports both new 'voice' and legacy 'azure-speech').
    
    Environment variables (AZURE_SPEECH_*) can override config.yaml values:
    - AZURE_SPEECH_KEY: Override the key
    - AZURE_SPEECH_REGION: Override the region
    - AZURE_SPEECH_VOICE: Override default voice
    - AZURE_SPEECH_STYLE: Override default style
    
    Args:
        config: The full config dictionary from config.yaml
        
    Returns:
        VoiceConfig if voice section exists and has region, None otherwise
    """
    global _azure_speech_warned
    
    # Check for new 'voice' section first, then fall back to legacy 'azure-speech'
    voice_cfg = config.get("voice") or config.get("azure-speech")
    
    # Track which key was used for deprecation warning
    voice_key = "voice" if "voice" in config else "azure-speech"
    
    # Log deprecation warning once when legacy key is used
    if voice_key == "azure-speech" and not _azure_speech_warned:
        warnings.warn(
            "The 'azure-speech' config key is deprecated. "
            "Please migrate to the new 'voice' section. "
            "See: https://github.com/zzzprojects/gpt-discord-bot/blob/main/openspec/specs/config/spec.md",
            DeprecationWarning,
            stacklevel=2
        )
        _azure_speech_warned = True
    
    if voice_cfg is None:
        return None
    
    if not isinstance(voice_cfg, dict):
        return None
    
    azure_speech = voice_cfg
    if azure_speech is None:
        return None
    
    if not isinstance(azure_speech, dict):
        return None
    
    # Get region with env var fallback (AZURE_SPEECH_REGION)
    region = _get_with_fallback(azure_speech.get("region", ""), "AZURE_SPEECH_REGION")
    if not region:
        return None
    
    # Get key with env var fallback (AZURE_SPEECH_KEY)
    # Note: Allow empty key - is_configured will be False
    key = _get_with_fallback(azure_speech.get("key", ""), "AZURE_SPEECH_KEY")
    
    # Get optional fields with env var fallback
    endpoint = _get_with_fallback(azure_speech.get("endpoint", ""), "AZURE_SPEECH_ENDPOINT")
    endpoint = endpoint if endpoint else None
    default_voice = _get_with_fallback(azure_speech.get("default_voice", ""), "AZURE_SPEECH_VOICE")
    default_voice = default_voice if default_voice else None
    default_style = _get_with_fallback(azure_speech.get("default_style", ""), "AZURE_SPEECH_STYLE")
    default_style = default_style if default_style else None
    
    return VoiceConfig(
        key=key,
        region=region,
        endpoint=endpoint,
        default_voice=default_voice,
        default_style=default_style,
    )