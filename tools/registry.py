"""
Central registry of tool names -> OpenAI-compatible tool schemas.
Used by both Ollama (via services/ollama) and OpenAI/OpenRouter (via llmcord).
Config references tools by name, e.g. tools: ["web_search", "web_fetch", "visuals_core"].
"""

from typing import Any, List

from tools.web_search import WEB_SEARCH_SCHEMA, WEB_FETCH_SCHEMA
from tools.visuals_core import VISUALS_CORE_SCHEMA

# Map config tool name -> full schema with type + function (OpenAI/OpenRouter format)
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "web_search": WEB_SEARCH_SCHEMA,
    "web_fetch": WEB_FETCH_SCHEMA,
    "visuals_core": VISUALS_CORE_SCHEMA,
}


def get_openai_tools(tool_names: List[str] | None) -> List[dict[str, Any]]:
    """
    Return a list of OpenAI-format tool dicts for the given names.
    Used for chat.completions.create(tools=...) with OpenAI/OpenRouter.
    """
    if not tool_names:
        return []
    return [TOOL_SCHEMAS[n] for n in tool_names if n in TOOL_SCHEMAS]
