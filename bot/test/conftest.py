"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests to avoid test pollution."""
    # Reset _warned_deprecated_keys in config loader
    import bot.config.loader as loader
    loader._warned_deprecated_keys.clear()
    
    # Reset _azure_speech_warned in voice config
    import bot.voice.config as voice_config
    voice_config._azure_speech_warned = False
    
    yield