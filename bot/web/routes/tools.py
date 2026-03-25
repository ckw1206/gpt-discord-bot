"""Tools API endpoints for getting available tools from the registry."""

import logging
from typing import Any, List

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tools"])


def _get_tool_schemas() -> List[dict[str, Any]]:
    """
    Get tool schemas from the registry.
    Returns list of {name, description, parameters} for each tool.
    """
    try:
        from bot.llm.tools import get_tools
        
        registry = get_tools()
        tools = []
        
        for name, entry in registry.items():
            # Extract description from schema if available
            description = ""
            if entry.schema:
                # Try to get description from schema
                schema = entry.schema
                if isinstance(schema, dict):
                    description = schema.get("description", "")
            
            # Extract parameter names from schema
            parameters = []
            if entry.schema and isinstance(entry.schema, dict):
                properties = entry.schema.get("parameters", {}).get("properties", {})
                for param_name, param_info in properties.items():
                    parameters.append({
                        "name": param_name,
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", "")
                    })
            
            tools.append({
                "name": name,
                "description": description,
                "parameters": parameters
            })
        
        return tools
    except Exception as e:
        logger.error(f"Failed to get tools from registry: {e}")
        return []


@router.get("/tools")
async def get_tools_endpoint() -> List[dict[str, Any]]:
    """
    Get available tools from the registry.
    
    Returns list of tools with their names, descriptions, and parameters.
    This endpoint returns actual tool names (e.g., 'get_market_prices')
    rather than skill file names (e.g., 'yahoo_finance').
    """
    return _get_tool_schemas()


@router.get("/tools/names")
async def get_tool_names() -> List[str]:
    """
    Get just the tool names (without full schema).
    Useful for simple dropdown lists.
    """
    try:
        from bot.llm.tools import get_tools
        registry = get_tools()
        return list(registry.keys())
    except Exception as e:
        logger.error(f"Failed to get tool names: {e}")
        return []