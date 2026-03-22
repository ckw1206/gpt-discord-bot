"""Config API endpoints for reading and updating configuration."""

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from bot.config.loader import get_config as get_raw_config, get_config_path
from bot.web.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["config"])


# Fields that are considered sensitive (never exposed via API)
SENSITIVE_FIELDS = {
    "api_key", "bot_token", "password", "secret", "key",
    "credentials", "token"
}

# Fields that are editable via the API
# Only new grouped keys (legacy keys auto-convert on load via normalization)
# Legacy keys are for backward-compatible LOADING only, not editing
EDITABLE_FIELDS = {
    # Discord section
    "discord.status_message", "discord.client_id", "discord.bot_token",
    # Behavior section
    "behavior.max_text", "behavior.max_images",
    "behavior.max_messages", "behavior.use_plain_responses",
    "behavior.show_embed_color", "behavior.allow_dms",
    # LLM section
    "llm.persona", "llm.system_prompt", "llm.fallback_models",
    # Voice section
    "voice.region", "voice.default_voice", "voice.key",
    # Portal section (nested fields)
    "portal.enabled", "portal.port", "portal.cors_origins", "portal.require_discord_admin",
    "portal.logs.retention_days", "portal.logs.levels",
}


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field name is considered sensitive."""
    field_lower = field_name.lower()
    return any(sensitive in field_lower for sensitive in SENSITIVE_FIELDS)


def _filter_config(config: dict[str, Any], redact_sensitive: bool = True) -> dict[str, Any]:
    """
    Filter config to remove sensitive fields and prepare for API response.
    Handles both new grouped structure and legacy flat keys.
    
    Args:
        config: The raw config dictionary
        redact_sensitive: If True, redact sensitive values; if False, just structure
    """
    result = {}
    
    for key, value in config.items():
        # Skip the entire portal section - it's handled separately
        if key == "portal" and redact_sensitive:
            continue
        
        # Handle legacy 'azure-speech' key - map to 'voice' for display
        if key == "azure-speech":
            key = "voice"
            
        if isinstance(value, dict):
            # Recursively filter nested dicts
            filtered = _filter_config(value, redact_sensitive)
            if filtered:  # Only include non-empty dicts
                result[key] = filtered
        elif redact_sensitive:
            # Redact sensitive values
            if _is_sensitive_field(key):
                result[key] = "[REDACTED]"
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result


def _format_for_display(config: dict[str, Any]) -> dict[str, Any]:
    """
    Format complex config fields for display-friendly presentation.
    Converts nested dicts to readable strings for UI display.
    
    Args:
        config: The config dictionary to format
    """
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            if key == "models":
                # Format models: list model names with their configs
                model_names = list(value.keys())
                result[key] = ", ".join(model_names) if model_names else "None"
            elif key == "permissions":
                # Format permissions: show top-level keys
                perm_keys = list(value.keys())
                result[key] = ", ".join(perm_keys) if perm_keys else "None"
            elif key == "providers":
                # Format providers: list provider names
                provider_names = list(value.keys())
                result[key] = ", ".join(provider_names) if provider_names else "None"
            elif key == "tools":
                # Format tools: show tool names and their enabled status
                tool_items = []
                for tool_name, tool_config in value.items():
                    if isinstance(tool_config, dict):
                        enabled = tool_config.get("enabled", False)
                        status = "✓" if enabled else "✗"
                        tool_items.append(f"{tool_name} ({status})")
                    else:
                        tool_items.append(str(tool_name))
                result[key] = ", ".join(tool_items) if tool_items else "None"
            elif key in ("voice", "azure-speech"):
                # Format voice config: show voice config keys
                voice_keys = list(value.keys()) if value else []
                result[key] = ", ".join(voice_keys) if voice_keys else "None"
            elif key == "fallback_models":
                # Format fallback_models: list model names
                fallback_list = list(value.keys()) if value else []
                result[key] = ", ".join(fallback_list) if fallback_list else "None"
            else:
                # For other dicts, recursively format
                result[key] = _format_for_display(value)
        elif isinstance(value, list):
            # Format lists to comma-separated strings
            if value and isinstance(value[0], dict):
                # List of dicts - show keys
                result[key] = ", ".join(str(v) for v in value)
            else:
                result[key] = ", ".join(str(v) for v in value) if value else "None"
        else:
            result[key] = value
    
    return result


def _get_portal_config_safe() -> dict[str, Any]:
    """Get portal config for API (excludes sensitive fields)."""
    try:
        config = get_raw_config()
        # Support both new 'portal' and legacy top-level portal keys
        portal = config.get("portal", {})
        
        # Extract safe fields
        return {
            "enabled": portal.get("enabled", False),
            "port": portal.get("port", 8080),
            "cors_origins": portal.get("cors_origins", []),
            "logs": {
                "retention_days": portal.get("logs", {}).get("retention_days", 7),
                "levels": portal.get("logs", {}).get("levels", ["INFO", "WARNING", "ERROR"]),
            },
            "require_discord_admin": portal.get("require_discord_admin", False),
        }
    except Exception as e:
        logger.error(f"Error loading portal config: {e}")
        return {}


class ConfigUpdateRequest(BaseModel):
    """Request model for updating config values."""
    status_message: Optional[str] = None
    max_text: Optional[int] = None
    max_images: Optional[int] = None
    max_messages: Optional[int] = None
    use_plain_responses: Optional[bool] = None
    show_embed_color: Optional[bool] = None
    allow_dms: Optional[bool] = None
    portal: Optional[dict[str, Any]] = None
    # Support for array-based updates from frontend
    fields: Optional[list[dict[str, Any]]] = None


class ConfigResponse(BaseModel):
    """Response model for config read."""
    config: dict[str, Any]
    portal: dict[str, Any]
    voice: Optional[dict[str, Any]] = None
    editable_fields: list[str]
    read_only_fields: list[str]
    # Structured LLM data for table editors (Task 9.2.7)
    llm_providers: Optional[dict[str, Any]] = None
    llm_models: Optional[dict[str, Any]] = None
    # Discord config (for bot_token visibility toggle)
    discord_config: Optional[dict[str, Any]] = None
    # Full config for JSON mode (entire config.yaml with sensitive fields redacted)
    full_config: Optional[dict[str, Any]] = None


class ConfigUpdateResponse(BaseModel):
    """Response for config update."""
    success: bool
    message: str
    reloaded_config: Optional[dict[str, Any]] = None


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get current configuration (filtered - sensitive fields redacted).
    
    Returns the full config with sensitive fields like api_key, bot_token
    replaced with "[REDACTED]". Complex nested fields (models, providers, etc.)
    are formatted for display-friendly presentation.
    """
    try:
        raw_config = get_raw_config()
        
        # Get filtered config (redacted)
        filtered = _filter_config(raw_config, redact_sensitive=True)
        
        # Format complex nested fields for display
        display_config = _format_for_display(filtered)
        
        # Get portal config separately
        portal_config = _get_portal_config_safe()
        
        # Get structured LLM data for table editors (Task 9.2.7)
        # Extract providers and models with full structure (NOT redacted - needed for UI toggle)
        llm_config = raw_config.get("llm", {})
        llm_providers = llm_config.get("providers", {})
        llm_models = llm_config.get("models", {})
        
        # Get discord config separately (for bot_token visibility)
        discord_config = raw_config.get("discord", {})
        
        # Get voice config separately (for key visibility toggle)
        voice_config = raw_config.get("voice", {})
        
        return ConfigResponse(
            config=display_config,
            portal=portal_config,
            voice=voice_config,  # Add voice config to response
            editable_fields=sorted(list(EDITABLE_FIELDS)),
            read_only_fields=sorted([
                "providers", "models", "fallback_models", "tools",
                "permissions"
            ]),
            llm_providers=llm_providers,
            llm_models=llm_models,
            discord_config=discord_config,
            full_config=filtered,  # Full config for JSON mode display
        )
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


@router.put("/config", response_model=ConfigUpdateResponse)
async def update_config(
    update: ConfigUpdateRequest,
    current_user: Any = Depends(get_current_user),
) -> ConfigUpdateResponse:
    """
    Update configuration values.
    
    Only a limited set of fields are editable:
    - status_message, max_text, max_images, max_messages
    - use_plain_responses, show_embed_color, allow_dms
    - portal (enabled, port, logs levels)
    
    Updates are written directly to config.yaml.
    """
    import yaml
    
    config_path = get_config_path()
    
    try:
        # Load current config
        with open(config_path, encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        # Apply updates
        updates_applied = []
        
        # Fields that should be saved as lists
        LIST_FIELDS = {"llm.fallback_models"}
        
        # Handle array-based updates from frontend
        if update.fields:
            for field in update.fields:
                key = field.get('key')
                value = field.get('value')
                if key and value is not None:
                    # Set nested key (e.g., "portal.enabled")
                    parts = key.split('.')
                    current = current_config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Handle list-type fields: convert comma-separated string to list
                    if key in LIST_FIELDS and isinstance(value, str):
                        # Frontend sends comma-separated string, convert to list
                        value = [s.strip() for s in value.split(',') if s.strip()]
                    
                    current[parts[-1]] = value
                    updates_applied.append(key)
        
        if update.status_message is not None:
            current_config["status_message"] = update.status_message
            updates_applied.append("status_message")
        
        if update.max_text is not None:
            current_config["max_text"] = update.max_text
            updates_applied.append("max_text")
        
        if update.max_images is not None:
            current_config["max_images"] = update.max_images
            updates_applied.append("max_images")
        
        if update.max_messages is not None:
            current_config["max_messages"] = update.max_messages
            updates_applied.append("max_messages")
        
        if update.use_plain_responses is not None:
            current_config["use_plain_responses"] = update.use_plain_responses
            updates_applied.append("use_plain_responses")
        
        if update.show_embed_color is not None:
            current_config["show_embed_color"] = update.show_embed_color
            updates_applied.append("show_embed_color")
        
        if update.allow_dms is not None:
            current_config["allow_dms"] = update.allow_dms
            updates_applied.append("allow_dms")
        
        # Handle portal updates
        if update.portal is not None:
            if "portal" not in current_config:
                current_config["portal"] = {}
            
            portal_updates = update.portal
            if "enabled" in portal_updates:
                current_config["portal"]["enabled"] = portal_updates["enabled"]
                updates_applied.append("portal.enabled")
            if "port" in portal_updates:
                current_config["portal"]["port"] = portal_updates["port"]
                updates_applied.append("portal.port")
            if "logs" in portal_updates:
                if "logs" not in current_config["portal"]:
                    current_config["portal"]["logs"] = {}
                if "retention_days" in portal_updates["logs"]:
                    current_config["portal"]["logs"]["retention_days"] = portal_updates["logs"]["retention_days"]
                    updates_applied.append("portal.logs.retention_days")
                if "levels" in portal_updates["logs"]:
                    current_config["portal"]["logs"]["levels"] = portal_updates["logs"]["levels"]
                    updates_applied.append("portal.logs.levels")
            if "require_discord_admin" in portal_updates:
                current_config["portal"]["require_discord_admin"] = portal_updates["require_discord_admin"]
                updates_applied.append("portal.require_discord_admin")
        
        # Write updated config back to file
        # Use allow_unicode=True to properly preserve Chinese/non-ASCII characters
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(current_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info(f"Config updated by user {current_user.username}: {updates_applied}")
        
        # Reload config to return updated values
        from bot.config import loader
        # Clear any cached config - force reload
        # Note: The config loader doesn't expose a clear function, 
        # so we'll just return the filtered current config
        reloaded = _filter_config(current_config, redact_sensitive=True)
        portal_config = _get_portal_config_safe()
        
        return ConfigUpdateResponse(
            success=True,
            message=f"Updated fields: {', '.join(updates_applied) if updates_applied else 'none'}",
            reloaded_config={**reloaded, "portal": portal_config},
        )
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


class ProviderUpdateRequest(BaseModel):
    """Request model for provider CRUD operations."""
    action: str  # "add", "update", "delete"
    name: str
    data: Optional[dict[str, Any]] = None


class ModelUpdateRequest(BaseModel):
    """Request model for model CRUD operations."""
    action: str  # "add", "update", "delete"
    model_name: str
    data: Optional[dict[str, Any]] = None


@router.put("/config/providers", response_model=ConfigUpdateResponse)
async def update_provider(
    update: ProviderUpdateRequest,
    current_user: Any = Depends(get_current_user),
) -> ConfigUpdateResponse:
    """
    CRUD operations for LLM providers.
    
    - add: Add new provider (requires name and data)
    - update: Update existing provider (requires name and data)
    - delete: Delete provider (requires name only)
    """
    import yaml
    
    config_path = get_config_path()
    
    try:
        # Load current config
        with open(config_path, encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        # Ensure llm.providers exists
        if "llm" not in current_config:
            current_config["llm"] = {}
        if "providers" not in current_config["llm"]:
            current_config["llm"]["providers"] = {}
        
        providers = current_config["llm"]["providers"]
        action = update.action.lower()
        name = update.name
        
        if action == "add":
            if name in providers:
                raise HTTPException(status_code=400, detail=f"Provider '{name}' already exists")
            if not update.data:
                raise HTTPException(status_code=400, detail="data required for add action")
            providers[name] = update.data
            message = f"Provider '{name}' added"
            
        elif action == "update":
            if name not in providers:
                raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
            if not update.data:
                raise HTTPException(status_code=400, detail="data required for update action")
            providers[name] = update.data
            message = f"Provider '{name}' updated"
            
        elif action == "delete":
            if name not in providers:
                raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
            del providers[name]
            message = f"Provider '{name}' deleted"
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}. Use add, update, or delete")
        
        # Write updated config back to file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(current_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info(f"Provider {action} by user {current_user.username}: {name}")
        
        return ConfigUpdateResponse(
            success=True,
            message=message,
            reloaded_config={"llm": {"providers": providers}},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating provider: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update provider: {str(e)}")


@router.put("/config/models", response_model=ConfigUpdateResponse)
async def update_model(
    update: ModelUpdateRequest,
    current_user: Any = Depends(get_current_user),
) -> ConfigUpdateResponse:
    """
    CRUD operations for LLM models.
    
    - add: Add new model (requires model_name and data)
    - update: Update existing model (requires model_name and data)
    - delete: Delete model (requires model_name only)
    """
    import yaml
    
    config_path = get_config_path()
    
    try:
        # Load current config
        with open(config_path, encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        # Ensure llm.models exists
        if "llm" not in current_config:
            current_config["llm"] = {}
        if "models" not in current_config["llm"]:
            current_config["llm"]["models"] = {}
        
        models = current_config["llm"]["models"]
        action = update.action.lower()
        model_name = update.model_name
        
        if action == "add":
            if model_name in models:
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' already exists")
            if not update.data:
                raise HTTPException(status_code=400, detail="data required for add action")
            models[model_name] = update.data
            message = f"Model '{model_name}' added"
            
        elif action == "update":
            if model_name not in models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            if not update.data:
                raise HTTPException(status_code=400, detail="data required for update action")
            models[model_name] = update.data
            message = f"Model '{model_name}' updated"
            
        elif action == "delete":
            if model_name not in models:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            del models[model_name]
            message = f"Model '{model_name}' deleted"
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}. Use add, update, or delete")
        
        # Write updated config back to file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(current_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info(f"Model {action} by user {current_user.username}: {model_name}")
        
        return ConfigUpdateResponse(
            success=True,
            message=message,
            reloaded_config={"llm": {"models": models}},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


@router.post("/refresh", response_model=ConfigUpdateResponse)
async def refresh_config(current_user: Any = Depends(get_current_user)) -> ConfigUpdateResponse:
    """
    Refresh/reload configuration from config.yaml.
    
    This endpoint allows reloading the config without restarting the bot.
    """
    try:
        # Force reload by re-reading from disk
        from bot.web.config import reload_portal_config
        from bot.web.routes.status import reload_bot_config, update_bot_presence
        
        # Reload portal config
        portal_config = reload_portal_config()
        
        # Also reload main bot config
        bot_config_reloaded = await reload_bot_config()
        
        # Update bot presence to reflect any status_message change (Task 16.5)
        presence_result = await update_bot_presence()
        
        logger.info(f"Config refreshed by user {current_user.username}")
        
        return ConfigUpdateResponse(
            success=True,
            message="Configuration reloaded successfully",
            reloaded_config={
                "portal": {
                    "enabled": portal_config.enabled,
                    "port": portal_config.port,
                    "logs": {
                        "retention_days": portal_config.logs_retention_days,
                        "levels": portal_config.logs_levels,
                    },
                },
                "bot": {
                    "reloaded": bot_config_reloaded,
                    "presence_updated": presence_result.get("success", False),
                }
            },
        )
    except Exception as e:
        logger.error(f"Error refreshing config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh config: {str(e)}")