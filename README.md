## gpt-discord-bot (fork)

Discord bot that connects to multiple LLM providers (OpenRouter, Ollama, Open WebUI, etc.), supports tools, personas (人格設定檔), and scheduled tasks.

### Layout

- `llmcord.py` – main entrypoint (Discord bot, message handling, scheduling).
- `bot/` – new modular package:
  - `bot/config/loader.py` – config loading (`CONFIG_PATH` env var supported).
  - `bot/config/personas/` – persona files (e.g. `lao_pi.md`, `stock_market_analyst.md`).
  - `bot/config/tasks/` – scheduled task definitions (`*.yaml`).
  - `bot/llm/` – LLM helpers, Ollama service, tools and error helpers.
  - `bot/discord/errors.py` – centralized Discord/admin error handling.

You can still run the bot exactly as before:

```bash
python llmcord.py
```

Or, via the new package entrypoint:

```bash
python -m bot.main
```

### Config basics (`config.yaml`)

- **Providers**: under `providers`, one key per provider (`openrouter`, `ollama`, `open-webui`, etc.).
- **Models**: under `models`, keys are `"provider/model-name"` (e.g. `openrouter/openrouter/free`).
- **Personas**:
  - Global: `persona: some_name` at top-level (optional).
  - Model-level: `persona: some_name` under a model.
  - Task-level: `persona: some_name` inside a scheduled task.
  - `some_name` maps to a file under `bot/config/personas/some_name.(md|txt|yaml|yml)`.
- **Fallback models**:
  - Global: `fallback_models: [...]` at top-level (used when a primary model fails).
  - Model-level: `fallback_models: [...]` under a model (used for interactive messages).
  - Task-level: `fallback_models: [...]` inside a task (used for that task only).

Resolution order:

- Interactive messages: `model.fallback_models` then global `fallback_models`.
- Scheduled tasks: task `fallback_models` then global `fallback_models`.

### Personas & Scheduled Tasks

For detailed instructions on creating and configuring **personas** and **scheduled tasks**, see [`bot/config/README.md`](bot/config/README.md).

Quick reference:
- **Personas**: Place files in `bot/config/personas/` (`.md`, `.txt`, `.yaml`)
- **Tasks**: Place files in `bot/config/tasks/` (`.yaml`)
- Tasks are auto-loaded and merged with `config.yaml` entries (file-based tasks take priority)

### Error handling

- All serious errors are sent to admins listed in `permissions.users.admin_ids`.
- For LLM/API errors:
  - The bot retries using configured fallback models.
  - If all models fail, users receive a short message (in zh-TW) and admins get a DM with details.
- For scheduled tasks:
  - Similar fallback behavior; if everything fails, the target channel/user is notified and admins are DM’d.
- Slash command errors are handled centrally and reported to admins via `bot/discord/errors.py`.

<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

**Note:** This is a fork of [llmcord](https://github.com/jakobdylanc/llmcord) with added scheduler support for automatic recurring tasks.

## Features

### Reply-based chat system:
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

---

### Scheduled tasks:
Configure periodic tasks to run on a schedule. Perfect for automated email checks, daily summaries, or any recurring LLM task.

- Define multiple tasks with different schedules (cron format)
- Send results to Discord channels or DMs
- Customize prompts and models per task
- Enable/disable tasks without removing configuration

---

### Model switching with `/model`:
![image](https://github.com/user-attachments/assets/568e2f5c-bf32-4b77-ab57-198d9120f3d2)

llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [xAI API](https://docs.x.ai/docs/models)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs/models)
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/models)

Or run local models with:
- [Ollama](https://ollama.com)
- [LM Studio](https://lmstudio.ai)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI compatible API server.

---

### Clear conversation history with `/clear`:
Use `/clear` to reset the conversation history and message cache. This is useful when:
- The bot seems stuck with warning messages
- You shared an image but then switched to a non-vision model
- You want to start a completely fresh conversation without context from previous messages

---

### And more:
- Supports image attachments when using a vision model (like gpt-5, grok-4, claude-4, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- User identity aware (OpenAI API and xAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Automatically strips thinking tags (`<think>` blocks) from reasoning models so users only see the final response
- Optional colored embed bars on responses (`show_embed_color` config option)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/ckw1206/gpt-discord-bot
   cd gpt-discord-bot
   ```

2. Create a copy of "config.yaml.default" named "config.yaml" and set it up:

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br /><br />**Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br /><br />Default: `100,000` |
| **max_images** | The maximum number of image attachments allowed in a single message.<br /><br />Default: `5`<br /><br />**Only applicable when using a vision model.** |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br /><br />Default: `25` |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often.<br /><br />Default: `false`<br /><br />**Also disables streamed responses and warning messages.** |
| **show_embed_color** | Show a green/orange colored bar on bot responses indicating completion status. Set to `false` to disable colored embeds and use plain responses instead.<br /><br />Default: `true` |
| **allow_dms** | Set to `false` to disable direct message access.<br /><br />Default: `true` |
| **permissions** | Configure access permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br /><br />Control which `users` are admins with `admin_ids`. Admins can change the model with `/model` and DM the bot even if `allow_dms` is `false`.<br /><br />**Leave `allowed_ids` empty to allow ALL in that category.**<br /><br />**Role and channel permissions do not affect DMs.**<br /><br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. Popular providers (`openai`, `openrouter`, `ollama`, etc.) are already included.<br /><br />**Only supports OpenAI compatible APIs.**<br /><br />**Some providers may need `extra_headers` / `extra_query` / `extra_body` entries for extra HTTP data. See the included `azure-openai` provider for an example.** |
| **models** | Add the models you want to use in `<provider>/<model>: <parameters>` format (examples are included). When you run `/model` these models will show up as autocomplete suggestions.<br /><br />**Refer to each provider's documentation for supported parameters.**<br /><br />**The first model in your `models` list will be the default model at startup.**<br /><br />**Some vision models may need `:vision` added to the end of their name to enable image support.** |
| **fallback_models** | *(Optional)* A list of models to automatically try if the primary model fails (e.g., rate limiting, API errors). The bot will try each fallback in order until one succeeds. If all models fail, the admin will be notified.<br /><br />**Format:** List of model names (matching the `models` section).<br /><br />**Example:**<br />```yaml<br />fallback_models:<br />  - "groq/mixtral-8x7b-32768"<br />  - "ollama/llama2"<br />```<br /><br />**Benefits:**<br />- Automatic failover when main model is rate-limited or unavailable<br />- No admin notifications if fallback succeeds<br />- Perfect for free-tier APIs with rate limits |
| **system_prompt** | *(Optional)* Customize the bot's behavior with a custom system prompt. Leave empty, blank, or remove entirely to use the model's default system prompt.<br /><br />**You can use the `{date}` and `{time}` tags in your system prompt to insert the current date and time, based on your host computer's time zone.**<br /><br />**Example:** `system_prompt: "You are a helpful assistant"` |

3. Run the bot:

   **No Docker:**
   ```bash
   python -m pip install -U -r requirements.txt
   python llmcord.py
   ```

   **With Docker:**
   ```bash
   docker compose up
   ```

## Ollama Integration (NEW)

The bot now supports **Ollama provider with tool calling**!

### Quick Setup

Add to your `config.yaml`:

```yaml
providers:
  ollama:
    base_url: http://127.0.0.1:11434

models:
  ollama/qwen3:14b:
    tools: ["web_search", "web_fetch", "get_temperature"]
    system_prompt: "You are a helpful assistant"
    think: true  # Optional: for reasoning models
```

### Available Tools

- `web_search` - Search the internet
- `web_fetch` - Get web page content  
- `get_temperature` - Get weather data

### Usage

Works everywhere:
- **Discord messages** - Tools execute automatically when needed
- **Scheduled tasks** - Use `tools` and optional `think` per task when the task's model is Ollama (e.g. `ollama/qwen3:14b`). Task-level `tools` / `think` override the model config. System prompt can be set per task with `system_prompt`.
- **Reasoning models** - Enable `think: true` for deep reasoning

### Features

| Feature | Status |
|---------|--------|
| Ollama support | ✅ |
| Tool calling | ✅ |
| Per-model system prompts | ✅ |
| Thinking mode | ✅ |
| Backward compatible | ✅ |

### System Prompt Priority

**For Discord messages:**
1. Per-model `system_prompt` (highest priority)
2. Global `system_prompt` (fallback)
3. Model default (if neither specified)

**For scheduled tasks:**
1. Task-level `system_prompt` (highest)
2. Model-level `system_prompt`
3. Global `system_prompt`
4. Model default

## Notes

- **Fallback models:** The bot now supports automatic failover with `fallback_models`. If your primary model is rate-limited or unavailable, the bot will try each fallback model in order. No admin notification is sent if a fallback succeeds.<br /><br />**Global example:**<br />```yaml<br />fallback_models:<br/>  - "groq/mixtral"<br />  - "ollama/llama2"<br />```<br /><br />For Discord messages, the response will still stream in real-time using whichever model succeeds. For scheduled tasks, the result will be sent normally.

- **Ollama tools:** Ollama models support tool calling. Configure tools per-model in `config.yaml` under `models.ollama/model-name.tools`. For **scheduled tasks** using an Ollama model, you can set `tools` (and `think`) on the task to override the model config or enable tools for that task. Available tools: `web_search`, `web_fetch`, `visuals_core`. Tool results are automatically formatted and displayed in responses.

- If you're having issues, try the suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- **Thinking tags (`<think>` blocks):** If your model includes thinking/reasoning content (like Claude's extended thinking), the bot automatically strips these tags so users only see the final response. The thinking content is never displayed in Discord.

- **Embed color bars:** By default, bot responses show a green (complete) or orange (incomplete) colored bar. To disable this and use plain color embeds instead, set `show_embed_color: false` in your config.

- **Persistent "⚠️ Can't see images" warning:** If you shared an image but your current model is not a vision model (like `gpt-4`, `claude`, `gemini`, etc.), the bot can't process the image and shows a warning. Run `/clear` to clear the message cache and start fresh.

- **Scheduled task DM issues:** If you get "Cannot send messages to this user" error when using `user_id` for scheduled tasks, the issue is likely one of:
  - **User has DMs disabled from bots** - Go to Discord settings → Privacy & Safety → Allow Direct Messages from server members (toggle on)
  - **Bot is blocked** - The user blocked the bot, so unblock it
  - **DM not initiated yet** - **Start a DM with the bot first** (send the bot a message in DM), then the scheduled task will work
  - **Using wrong user_id** - Double-check the Discord user ID is correct
  - **As a workaround:** Use `channel_id` instead of `user_id` to send results to a Discord channel where the bot has permission

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Hopefully more providers support this in the future.

- This is a fork of [llmcord](https://github.com/jakobdylanc/llmcord) with scheduled task support. Check the original repo for more information.

- PRs are welcome :)