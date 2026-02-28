from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from ollama import Client

from .tools.web_search import format_web_search_results, WEB_SEARCH_SCHEMA, WEB_FETCH_SCHEMA
from .tools.visuals_core import generate_visualization, VISUALS_CORE_SCHEMA


class OllamaService:
    def __init__(self, host: str):
        load_dotenv()

        api_key = os.getenv("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = Client(host=host, headers=headers)

        # ── Callables: what actually runs when a tool is invoked ────────────
        def _web_search(query: str) -> Any:
            return self.client.web_search(query=query)

        def _web_fetch(url: str) -> Any:
            return self.client.web_fetch(url=url)

        def _visuals_core(viz_type: str, data: str, title: str = "") -> str:
            try:
                kwargs = json.loads(data) if isinstance(data, str) else data
            except Exception:
                kwargs = {}
            return generate_visualization(viz_type=viz_type, title=title, **kwargs)

        self.tool_map: Dict[str, Any] = {
            "web_search": _web_search,
            "web_fetch": _web_fetch,
            "visuals_core": _visuals_core,
        }

        # ── Schemas: what gets sent to the Ollama client ────────────────────
        self.tool_schemas: Dict[str, dict] = {
            "web_search": WEB_SEARCH_SCHEMA,
            "web_fetch": WEB_FETCH_SCHEMA,
            "visuals_core": VISUALS_CORE_SCHEMA,
        }

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
            self.tool_schemas[n] for n in (enable_tools or []) if n in self.tool_schemas
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
                fn = self.tool_map.get(name)

                if not fn:
                    logging.warning("OllamaService: unknown tool '%s'", name)
                    continue

                logging.info("OllamaService: tool '%s' args=%s", name, args)
                try:
                    result = fn(**args)
                except Exception as e:
                    logging.error("OllamaService: tool '%s' failed: %s", name, e)
                    result = f"Tool error: {e}"

                formatted = self._format(name, result, args)[:max_tool_chars]
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

    # ── Formatter ───────────────────────────────────────────────────────────

    def _format(self, name: str, result: Any, args: dict) -> str:
        if name == "web_search":
            return format_web_search_results(result, user_search=args.get("query", ""))
        if name == "web_fetch":
            return format_web_search_results(result, user_search=args.get("url", ""))
        return str(result)

