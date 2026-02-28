"""
You have access to two web tools: web_search and web_fetch.

Use web_search when:
- The user asks for up-to-date, recent, or real-time information.
- The answer may change over time (e.g., news, prices, releases, events).
- You need external information that is not guaranteed to be in your knowledge base.

When using web_search:
- Extract relevant facts directly from the search results.
- Do NOT summarize what the website does.
- Do NOT describe the website itself.
- Only include information directly related to the user's question.

Use web_fetch when:
- The user provides a specific URL.
- You need to retrieve detailed content from that page.

When using web_fetch:
- Only present the information requested by the user.
- Ignore irrelevant sections of the page.
- Do NOT describe the website itself.

Always prioritize factual extraction over explanation of the source.
"""

from dotenv import load_dotenv
from ollama import WebFetchResponse, WebSearchResponse
from typing import Union


def format_web_search_results(
    results: Union[WebSearchResponse, WebFetchResponse],
    user_search: str,
):
    output = []

    if isinstance(results, WebSearchResponse):
        output.append(f'Search results for "{user_search}":')
        for result in results.results:
            output.append(f'{result.title}' if result.title else f'{result.content}')
            output.append(f'   URL: {result.url}')
            output.append(f'   Content: {result.content}')
            output.append('')
        return '\n'.join(output).rstrip()

    elif isinstance(results, WebFetchResponse):
        output.append(f'Fetch results for "{user_search}":')
        output.extend([
            f'Title: {results.title}',
            f'URL: {user_search}' if user_search else '',
            f'Content: {results.content}',
        ])
        if results.links:
            output.append(f'Links: {", ".join(results.links)}')
        output.append('')
        return '\n'.join(output).rstrip()


# ── Ollama Tool Schemas ───────────────────────────────────────────────────────

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

WEB_FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch the contents of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
}