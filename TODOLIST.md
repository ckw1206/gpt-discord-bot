# TODO List

Backlog of improvements, bugs, and ideas for the bot.
Check items off as they're completed.

---

## ✅ Done

- [x] Ollama provider with tool calling (`web_search`, `web_fetch`, `visuals_core`)
- [x] Tool schemas decoupled from tool callables — each tool file owns both
- [x] `registry.py` as single source of truth for all tool schemas
- [x] `build_tool_registry(client)` — `ollama_service.py` has zero hardcoded tool names
- [x] `ToolEntry` dataclass (schema + fn + formatter) — uniform tool contract
- [x] `SKILLS.md` — human + AI reference for adding/using tools
- [x] Fallback model chain — primary → model fallbacks → global fallbacks
- [x] Per-model timeout (`RESPONSE_TIMEOUT_SECONDS`) with fallback on expiry
- [x] Graceful tool call parse error recovery (retries without tools on Ollama 500)
- [x] Persona system (`bot/config/personas/`)
- [x] Scheduled tasks with cron (`bot/config/tasks/`)
- [x] `/model` slash command with autocomplete
- [x] `/clear` slash command
- [x] Streamed responses with embed color (green = complete, orange = streaming)
- [x] `<think>` tag stripping for reasoning models
- [x] Admin error notifications via Discord DM

---

## 🔧 In Progress / Next Up

- [ ] **Tool result timeout** — individual tool calls (e.g. slow `web_fetch`) can hang;
      add per-tool timeout wrapping `entry.fn(**args)` in `OllamaService.run()`
- [ ] **`web_search` for OpenRouter** — current web_search only works with Ollama native.
      For OpenRouter, needs an external API (Tavily, Brave, SerpAPI) or a search-native
      model (e.g. `perplexity/sonar`). Add a second callable in `web_search.py` and
      detect provider in `registry.py` or `llmcord.py`.

---

## 💡 Ideas / Backlog

- [ ] **`/tools` slash command** — list available tools and their descriptions, sourced
      from `SKILLS.md` or `registry.py` at runtime
- [ ] **Hot-reload tools** — watch `registry.py` for changes and reload without restart
- [ ] **Tool usage logging** — log which tools were called per conversation to a file or
      Discord channel for debugging
- [ ] **`get_temperature` tool** — weather tool stub already referenced in old README;
      implement and register in `registry.py`
- [ ] **Per-task tool override in scheduled tasks** — tasks can already set `tools:` but
      there's no validation that the tool names exist; add a check in `bot/config/tasks.py`
- [ ] **Config validator for tools** — `bot/config/validator.py` should check that tool
      names in `config.yaml` match keys in `registry._ENTRIES`
- [ ] **Multi-turn tool calling for OpenAI/OpenRouter** — current OpenAI path streams
      responses but doesn't implement the tool call → result → continue loop that Ollama
      does. Would need a non-streaming first pass to detect tool calls.
- [ ] **Dockerfile improvements** — pin Python version, add health check
- [ ] **Tests for OllamaService** — `test_ollama.py` exists but is thin; add unit tests
      for the tool registry and format_tool_result

---

## 🐛 Known Issues

- `qwen3:14b` with `think: true` bleeds reasoning tokens into tool call JSON, causing
  Ollama to return HTTP 500. Workaround is already in place (retry without tools), but
  the model is fundamentally unreliable for tool calling. Use `qwen2.5:14b` instead.
- `web_search` / `web_fetch` don't work with OpenRouter — they're Ollama-native only.
  See "In Progress" above.
