"""
Unit tests for config loader normalization.

Tests that the config loader correctly handles:
1. Legacy flat config structure → new grouped structure
2. New grouped config structure (pass-through)
3. Mixed config (legacy + new) - new takes precedence
4. Backward compatibility aliases
"""

import pytest
from bot.config.loader import _normalize_config, LEGACY_KEY_MAPPING


class TestLegacyToGroupedConversion:
    """Tests for converting legacy flat config to new grouped structure."""
    
    def test_legacy_bot_token_converted_to_discord(self):
        """Legacy bot_token should be moved to discord.bot_token."""
        config = {
            "bot_token": "test_token_123",
            "providers": {"ollama": {"base_url": "http://localhost:11434"}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        assert "discord" in result
        assert result["discord"]["bot_token"] == "test_token_123"
        # Legacy key is kept as backward compatibility alias
        assert result["bot_token"] == "test_token_123"
    
    def test_legacy_client_id_converted_to_discord(self):
        """Legacy client_id should be moved to discord.client_id."""
        config = {
            "bot_token": "test_token",
            "client_id": "123456789",
            "providers": {"ollama": {}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        assert result["discord"]["client_id"] == "123456789"
        # Legacy key is kept as backward compatibility alias
        assert result["client_id"] == "123456789"
    
    def test_legacy_permissions_converted_to_discord(self):
        """Legacy permissions should be moved to discord.permissions."""
        config = {
            "bot_token": "test_token",
            "permissions": {"users": {"allowed_ids": [123]}},
            "providers": {"ollama": {}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        assert result["discord"]["permissions"]["users"]["allowed_ids"] == [123]
    
    def test_legacy_behavior_keys_converted(self):
        """Legacy behavior keys should be moved to behavior section."""
        config = {
            "bot_token": "test_token",
            "max_text": 50000,
            "max_images": 3,
            "max_messages": 10,
            "use_plain_responses": True,
            "providers": {"ollama": {}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        assert result["behavior"]["max_text"] == 50000
        assert result["behavior"]["max_images"] == 3
        assert result["behavior"]["max_messages"] == 10
        assert result["behavior"]["use_plain_responses"] is True
        # Original keys should be removed
        assert "max_text" not in result
    
    def test_legacy_llm_keys_converted(self):
        """Legacy llm keys should be moved to llm section."""
        config = {
            "bot_token": "test_token",
            "providers": {"ollama": {}},
            "models": {"test/model": {}},
            "fallback_models": ["model2"],
            "persona": "assistant",
            "system_prompt": "You are helpful."
        }
        result = _normalize_config(config)
        
        assert result["llm"]["providers"]["ollama"] == {}
        assert result["llm"]["models"]["test/model"] == {}
        assert result["llm"]["fallback_models"] == ["model2"]
        assert result["llm"]["persona"] == "assistant"
        assert result["llm"]["system_prompt"] == "You are helpful."
    
    def test_legacy_azure_speech_converted_to_voice(self):
        """Legacy azure-speech should be moved to voice section."""
        config = {
            "bot_token": "test_token",
            "azure-speech": {"key": "value"},
            "providers": {"ollama": {}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        assert "voice" in result
        assert result["voice"]["key"] == "value"
        assert "azure-speech" not in result


class TestNewGroupedStructure:
    """Tests for new grouped config structure (should pass through unchanged)."""
    
    def test_new_grouped_config_unchanged(self):
        """New grouped config should remain unchanged."""
        config = {
            "discord": {
                "bot_token": "test_token",
                "client_id": "123"
            },
            "behavior": {
                "max_text": 100000
            },
            "llm": {
                "providers": {"ollama": {}},
                "models": {"test/model": {}}
            }
        }
        result = _normalize_config(config)
        
        assert result["discord"]["bot_token"] == "test_token"
        assert result["discord"]["client_id"] == "123"
        assert result["behavior"]["max_text"] == 100000
        assert result["llm"]["providers"]["ollama"] == {}


class TestMixedConfigPrecedence:
    """Tests for mixed config where both legacy and new keys exist."""
    
    def test_new_takes_precedence_over_legacy(self):
        """When both legacy and new exist, new format takes precedence."""
        config = {
            "bot_token": "legacy_token",  # Will be moved
            "discord": {
                "bot_token": "new_token"  # Takes precedence
            },
            "providers": {"ollama": {}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        # Should use new token, not legacy
        assert result["discord"]["bot_token"] == "new_token"
        # Legacy token is kept as alias pointing to new value (backward compat)
        assert result["bot_token"] == "new_token"
    
    def test_partial_migration_works(self):
        """Partial migration (some legacy, some new) should work correctly."""
        config = {
            "bot_token": "legacy_token",  # Legacy
            "client_id": "123",           # Legacy
            "llm": {                      # New format
                "providers": {"ollama": {}},
                "models": {"test/model": {}}
            },
            "behavior": {                 # New format
                "max_text": 50000
            }
        }
        result = _normalize_config(config)
        
        # Legacy converted
        assert result["discord"]["bot_token"] == "legacy_token"
        assert result["discord"]["client_id"] == "123"
        # New preserved
        assert result["llm"]["providers"]["ollama"] == {}
        assert result["behavior"]["max_text"] == 50000


class TestBackwardCompatibilityAliases:
    """Tests for backward compatibility aliases."""
    
    def test_models_alias_available(self):
        """config['models'] should be available when using llm.models."""
        config = {
            "bot_token": "test",
            "llm": {
                "providers": {"ollama": {}},
                "models": {"test/model": {"persona": "test"}}
            }
        }
        result = _normalize_config(config)
        
        # Both llm.models and top-level models should work
        assert result["llm"]["models"]["test/model"]["persona"] == "test"
        assert result["models"]["test/model"]["persona"] == "test"
    
    def test_providers_alias_available(self):
        """config['providers'] should be available when using llm.providers."""
        config = {
            "bot_token": "test",
            "llm": {
                "providers": {"ollama": {"base_url": "http://localhost:11434"}},
                "models": {"test/model": {}}
            }
        }
        result = _normalize_config(config)
        
        # Both llm.providers and top-level providers should work
        assert result["llm"]["providers"]["ollama"]["base_url"] == "http://localhost:11434"
        assert result["providers"]["ollama"]["base_url"] == "http://localhost:11434"
    
    def test_discord_aliases_available(self):
        """config['bot_token'], config['client_id'], etc. should be available when using discord.*."""
        config = {
            "discord": {
                "bot_token": "test_token",
                "client_id": "123456",
                "status_message": "Hello",
                "permissions": {"users": {"allowed_ids": [111]}}
            },
            "llm": {
                "providers": {"ollama": {}},
                "models": {"test/model": {}}
            }
        }
        result = _normalize_config(config)
        
        # All discord keys should have top-level aliases
        assert result["bot_token"] == "test_token"
        assert result["client_id"] == "123456"
        assert result["status_message"] == "Hello"
        assert result["permissions"]["users"]["allowed_ids"] == [111]


class TestAzureSpeechMerging:
    """Tests for azure-speech to voice merging behavior."""
    
    def test_azure_speech_merged_with_voice(self):
        """If both azure-speech and voice exist, they should be merged."""
        config = {
            "bot_token": "test",
            "azure-speech": {"speech_key": "value1"},
            "voice": {"voice_key": "value2"},
            "providers": {"ollama": {}},
            "models": {"test/model": {}}
        }
        result = _normalize_config(config)
        
        # Both should be merged into voice
        assert result["voice"]["speech_key"] == "value1"
        assert result["voice"]["voice_key"] == "value2"


class TestLegacyKeyMapping:
    """Tests for the LEGACY_KEY_MAPPING constant."""
    
    def test_all_legacy_keys_mapped(self):
        """Verify all expected legacy keys are in the mapping."""
        expected_mappings = {
            "bot_token": ("discord", "bot_token"),
            "client_id": ("discord", "client_id"),
            "status_message": ("discord", "status_message"),
            "permissions": ("discord", "permissions"),
            "max_text": ("behavior", "max_text"),
            "max_images": ("behavior", "max_images"),
            "max_messages": ("behavior", "max_messages"),
            "use_plain_responses": ("behavior", "use_plain_responses"),
            "show_embed_color": ("behavior", "show_embed_color"),
            "allow_dms": ("behavior", "allow_dms"),
            "providers": ("llm", "providers"),
            "models": ("llm", "models"),
            "fallback_models": ("llm", "fallback_models"),
            "persona": ("llm", "persona"),
            "system_prompt": ("llm", "system_prompt"),
            "azure-speech": ("voice", None),  # None = move entire key
        }
        
        for key, expected in expected_mappings.items():
            assert key in LEGACY_KEY_MAPPING, f"Missing mapping for {key}"
            assert LEGACY_KEY_MAPPING[key] == expected, f"Wrong mapping for {key}"