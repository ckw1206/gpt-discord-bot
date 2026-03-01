"""
bot/llm/ollama_service.py

Ollama LLM runner with tool-calling support.
All tool definitions live in bot/llm/tools/registry.py — nothing is hardcoded here.
See SKILLS.md for how to add new tools.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from ollama import Client

from .tools.registry import build_tool_registry, format_tool_result


class OllamaService:
    def __init__(self, host: str):
        load_dotenv()

        api_key = os.getenv("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = Client(host=host, headers=headers)

        # All tool callables + schemas + formatters loaded from registry.
        # To add a new tool, edit bot/llm/tools/registry.py only.
        self._registry = build_tool_registry(self.client)

    # ── Main Chat Runner ────────────────────────────────────────────────────

    def run(
        self,
        messages: List[Dict[str, str]],
        model: str,
        enable_tools: List[str] | None = None,
        think: bool = False,
        max_tool_chars: int = 8000,
    ) -> Dict[str, Any]:

        enabled_schemas = [
            self._registry[n].schema
            for n in (enable_tools or [])
            if n in self._registry
        ]
        tool_outputs = []

        while True:
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    tools=enabled_schemas,
                    think=think,
                )
            except Exception as e:
                if enabled_schemas and "error parsing tool call" in str(e):
                    # Model doesn't support structured tool calls (e.g. qwen3 bleeding
                    # reasoning tokens into tool JSON). Retry without tools so the user
                    # at least gets a plain-text answer instead of a crash.
                    logging.warning(
                        "OllamaService: model produced unparseable tool call "
                        "(likely model incompatibility). Retrying without tools."
                    )
                    enabled_schemas = []
                    response = self.client.chat(
                        model=model, messages=messages, tools=[], think=think
                    )
                else:
                    raise

            messages.append(response.message)

            logging.info(
                "OllamaService: response content=%r, tool_calls=%s",
                response.message.content,
                bool(response.message.tool_calls),
            )
            if not response.message.tool_calls:
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments
                entry = self._registry.get(name)

                if not entry or not entry.fn:
                    logging.warning("OllamaService: unknown tool '%s'", name)
                    continue

                logging.info("OllamaService: tool '%s' args=%s", name, args)
                try:
                    result = entry.fn(**args)
                except Exception as e:
                    logging.error("OllamaService: tool '%s' failed: %s", name, e)
                    result = f"Tool error: {e}"

                formatted = format_tool_result(entry, result, args)[:max_tool_chars]
                tool_outputs.append(formatted)
                messages.append(
                    {"role": "tool", "tool_name": name, "content": formatted}
                )

        return {
            "content": response.message.content or "",
            "thinking": getattr(response.message, "thinking", None),
            "tool_results": tool_outputs,
            "messages": messages,
        }
