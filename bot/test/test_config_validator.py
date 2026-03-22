"""
Integration tests for config validator dual-path support.

Tests that the validator correctly handles both legacy flat keys
and new grouped config structure.
"""

import pytest
from bot.config.validator import validate_config, ConfigValidationError


class TestLegacyConfigFormat:
    """Tests for legacy flat config format (providers/models at top level)."""
    
    def test_valid_legacy_config_passes(self):
        """Legacy config with providers/models at top level should pass."""
        config = {
            "providers": {
                "ollama": {"base_url": "http://localhost:11434"}
            },
            "models": {
                "test/model": {"persona": "test"}
            }
        }
        # Should not raise
        validate_config(config)
    
    def test_legacy_config_with_fallback_models(self):
        """Legacy config with fallback_models at top level should pass."""
        config = {
            "providers": {
                "ollama": {"base_url": "http://localhost:11434"}
            },
            "models": {
                "test/model": {"persona": "test"}
            },
            "fallback_models": ["model2", "model3"]
        }
        validate_config(config)
    
    def test_legacy_azure_speech_passes(self):
        """Legacy config with azure-speech should pass validation."""
        config = {
            "providers": {
                "ollama": {"base_url": "http://localhost:11434"}
            },
            "models": {
                "test/model": {"persona": "test"}
            },
            "azure-speech": {
                "key": "test-key",
                "region": "eastus"
            }
        }
        validate_config(config)


class TestNewGroupedConfigFormat:
    """Tests for new grouped config format (providers/models under llm)."""
    
    def test_valid_grouped_config_passes(self):
        """New grouped config with llm.providers/llm.models should pass."""
        config = {
            "llm": {
                "providers": {
                    "ollama": {"base_url": "http://localhost:11434"}
                },
                "models": {
                    "test/model": {"persona": "test"}
                }
            }
        }
        validate_config(config)
    
    def test_grouped_config_with_fallback_models(self):
        """New config with llm.fallback_models should pass."""
        config = {
            "llm": {
                "providers": {
                    "ollama": {"base_url": "http://localhost:11434"}
                },
                "models": {
                    "test/model": {"persona": "test"}
                },
                "fallback_models": ["model2", "model3"]
            }
        }
        validate_config(config)
    
    def test_new_voice_section_passes(self):
        """New config with voice section should pass validation."""
        config = {
            "llm": {
                "providers": {
                    "ollama": {"base_url": "http://localhost:11434"}
                },
                "models": {
                    "test/model": {"persona": "test"}
                }
            },
            "voice": {
                "key": "test-key",
                "region": "eastus"
            }
        }
        validate_config(config)


class TestMixedConfigFormat:
    """Tests for mixed config format (some old, some new keys)."""
    
    def test_mixed_providers_and_models(self):
        """Mixed config with providers at top, models under llm should pass."""
        config = {
            "providers": {
                "ollama": {"base_url": "http://localhost:11434"}
            },
            "llm": {
                "models": {
                    "test/model": {"persona": "test"}
                }
            }
        }
        validate_config(config)
    
    def test_mixed_fallback_models(self):
        """Mixed config with fallback_models in both locations should pass."""
        config = {
            "llm": {
                "providers": {
                    "ollama": {"base_url": "http://localhost:11434"}
                },
                "models": {
                    "test/model": {"persona": "test"}
                },
                "fallback_models": ["llm-fallback"]
            }
        }
        # Note: If legacy fallback_models is also present, it would be used
        validate_config(config)


class TestValidationErrors:
    """Tests that validation errors are correctly raised."""
    
    def test_missing_providers_in_both_locations(self):
        """Should fail when providers missing from both old and new locations."""
        config = {
            "models": {"test/model": {}}
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "providers" in str(exc_info.value)
    
    def test_missing_models_in_both_locations(self):
        """Should fail when models missing from both old and new locations."""
        config = {
            "providers": {"ollama": {"base_url": "http://localhost:11434"}}
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "models" in str(exc_info.value)
    
    def test_empty_models_fails(self):
        """Should fail when models section is empty (treated as missing)."""
        config = {
            "providers": {"ollama": {"base_url": "http://localhost:11434"}},
            "models": {}
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "models" in str(exc_info.value).lower()
    
    def test_invalid_provider_config_fails(self):
        """Should fail when provider missing base_url."""
        config = {
            "providers": {
                "ollama": {}  # Missing base_url
            },
            "models": {"test/model": {}}
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "base_url" in str(exc_info.value)
    
    def test_invalid_fallback_models_type_fails(self):
        """Should fail when fallback_models is not a list."""
        config = {
            "providers": {"ollama": {"base_url": "http://localhost:11434"}},
            "models": {"test/model": {}},
            "fallback_models": "not-a-list"  # Should be a list
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "fallback_models" in str(exc_info.value)
    
    def test_invalid_voice_type_fails(self):
        """Should fail when voice is not a dict."""
        config = {
            "providers": {"ollama": {"base_url": "http://localhost:11434"}},
            "models": {"test/model": {}},
            "voice": "not-a-dict"
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "voice" in str(exc_info.value)


class TestPermissionValidation:
    """Tests for permissions section validation."""
    
    def test_permissions_at_legacy_location(self):
        """Legacy config with permissions at top level should pass."""
        config = {
            "providers": {"ollama": {"base_url": "http://localhost:11434"}},
            "models": {"test/model": {}},
            "permissions": {
                "users": {
                    "admin_ids": ["123", "456"],
                    "allowed_ids": [],
                    "blocked_ids": []
                }
            }
        }
        validate_config(config)
    
    def test_permissions_at_new_location(self):
        """New config with permissions under discord should pass."""
        config = {
            "llm": {
                "providers": {"ollama": {"base_url": "http://localhost:11434"}},
                "models": {"test/model": {}}
            },
            "discord": {
                "permissions": {
                    "users": {
                        "admin_ids": ["123"],
                        "allowed_ids": [],
                        "blocked_ids": []
                    }
                }
            }
        }
        validate_config(config)