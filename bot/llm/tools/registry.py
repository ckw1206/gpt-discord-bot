"""
bot/llm/tools/registry.py

Single source of truth for ALL bot tools.
Each ToolEntry bundles: the callable, the OpenAI-format schema, and an optional formatter.

Adding a new tool only requires:
  1. Create bot/llm/tools/my_tool.py  (fn + SCHEMA)
  2. Import and add a ToolEntry below in _ENTRIES
  3. Add the tool name to config.yaml under the model's `tools:` list

See SKILLS.md at the repo root for full documentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, List

from .web_search import (
    WEB_FETCH_SCHEMA,
    WEB_SEARCH_SCHEMA,
    format_web_search_results,
)
from .visuals_core import VISUALS_CORE_SCHEMA, generate_visualization


# ── ToolEntry ─────────────────────────────────────────────────────────────────

@dataclass
class ToolEntry:
    schema: dict                       # OpenAI-format schema sent to the model
    fn: Callable | None = None         # called locally when model invokes this tool
    formatter: Callable | None = None  # optional: formatter(result, args) -> str


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_web_search(result: Any, args: dict) -> str:
    return format_web_search_results(result, user_search=args.get("query", ""))

def _fmt_web_fetch(result: Any, args: dict) -> str:
    return format_web_search_results(result, user_search=args.get("url", ""))


# ── visuals_core wrapper ───────────────────────────────────────────────────────
# Unpacks the JSON `data` string arg before calling generate_visualization,
# because Ollama models pass nested dicts as JSON strings.

def _visuals_core(viz_type: str, data: str, title: str = "") -> str:
    try:
        kwargs = json.loads(data) if isinstance(data, str) else data
    except Exception:
        kwargs = {}
    return generate_visualization(viz_type=viz_type, title=title, **kwargs)


# ── Tool definitions ──────────────────────────────────────────────────────────
# fn for web_search / web_fetch is None here; it's bound to an Ollama client
# at runtime by build_tool_registry(client). All other tools are static.
#
# To add a new tool, append an entry here. That's the only file to touch.

_ENTRIES: dict[str, ToolEntry] = {
    "web_search": ToolEntry(
        schema=WEB_SEARCH_SCHEMA,
        fn=None,                    # bound at runtime in build_tool_registry()
        formatter=_fmt_web_search,
    ),
    "web_fetch": ToolEntry(
        schema=WEB_FETCH_SCHEMA,
        fn=None,                    # bound at runtime in build_tool_registry()
        formatter=_fmt_web_fetch,
    ),
    "visuals_core": ToolEntry(
        schema=VISUALS_CORE_SCHEMA,
        fn=_visuals_core,
    ),
}


# ── Ollama runtime registry ───────────────────────────────────────────────────

def build_tool_registry(ollama_client: Any) -> dict[str, ToolEntry]:
    """
    Return a fully-wired registry for a given Ollama Client instance.
    Ollama-native tools (web_search, web_fetch) get their fn bound here.
    Call once per OllamaService.__init__.
    """
    registry = dict(_ENTRIES)
    registry["web_search"] = ToolEntry(
        schema=WEB_SEARCH_SCHEMA,
        fn=lambda query: ollama_client.web_search(query=query),
        formatter=_fmt_web_search,
    )
    registry["web_fetch"] = ToolEntry(
        schema=WEB_FETCH_SCHEMA,
        fn=lambda url: ollama_client.web_fetch(url=url),
        formatter=_fmt_web_fetch,
    )
    return registry


# ── OpenAI / OpenRouter helper ────────────────────────────────────────────────

def get_openai_tools(tool_names: List[str] | None) -> List[dict[str, Any]]:
    """
    Return OpenAI-format schema dicts for the given tool names.
    Used for chat.completions.create(tools=...) with OpenAI/OpenRouter.
    """
    if not tool_names:
        return []
    return [_ENTRIES[n].schema for n in tool_names if n in _ENTRIES]


# ── Result formatter ──────────────────────────────────────────────────────────

def format_tool_result(entry: ToolEntry, result: Any, args: dict) -> str:
    """Format a tool result via the entry's formatter, or fall back to str()."""
    if entry.formatter:
        try:
            return entry.formatter(result, args)
        except Exception as e:
            logging.warning("Tool formatter failed: %s", e)
    return str(result)
