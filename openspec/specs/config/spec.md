# Config Spec

## Purpose
Defines the configuration file structure for the Discord bot, including providers, models, personas, tasks, and permissions.

## Config Structure

The bot supports **two config formats**: the new **grouped structure** (recommended) and the legacy **flat structure** (deprecated but supported for backward compatibility).

### New Grouped Structure (Recommended)

```yaml
discord:
  bot_token: <token>
  client_id: <id>
  status_message: <message>
  permissions: {...}

behavior:
  max_text: 100000
  max_images: 5
  max_messages: 25
  use_plain_responses: false
  show_embed_color: true
  allow_dms: true

llm:
  providers: {...}
  models: {...}
  fallback_models: [...]
  persona: <name>
  system_prompt: <prompt>

tools: {...}
voice: {...}
portal: {...}
scheduled_tasks: {...}
```

### Legacy Flat Structure (Deprecated)

```yaml
bot_token: <token>
client_id: <id>
status_message: <message>
permissions: {...}
max_text: 100000
providers: {...}
models: {...}
...
```

The config loader automatically converts legacy keys to the new grouped structure. When both formats are used, the new grouped structure takes precedence.

## Requirements
### Requirement: Config schema defines required structure
The config SHALL have these keys (in either format):
- **discord.bot_token** (or `bot_token`): Discord bot token
- **discord.client_id** (or `client_id`): Discord client ID (optional)
- **llm.providers** (or `providers`): LLM provider configurations
- **llm.models** (or `models`): Model configurations with provider/model format
- Optional: `llm.persona`, `llm.system_prompt`, `llm.fallback_models`, `scheduled_tasks`, `discord.permissions`, `discord.status_message`, `behavior.*`

#### Scenario: Config structure
- **WHEN** AI reads config.yaml
- **THEN** it can parse: providers, models, bot_token, persona, system_prompt, fallback_models, scheduled_tasks, permissions (in either format)

### Requirement: Config validation ensures required fields
The system SHALL validate config and fail startup if required fields are missing.

#### Scenario: Validation
- **WHEN** config is loaded
- **THEN** validator.py checks: providers exists, models exists, each model has valid provider

### Requirement: Persona loaded from file
The system SHALL load personas from `bot/config/personas/<name>.(md|txt|yaml|yml)`.

#### Scenario: Load persona
- **WHEN** persona: bao is configured
- **THEN** loads bot/config/personas/bao.md

## Config File

Default: `config.yaml` (or path from CONFIG_PATH env)

## Top-Level Keys (Grouped Structure)

### discord Section
| Key | Type | Required | Description |
|-----|------|----------|-------------|
| discord.bot_token | string | Yes | Discord bot token |
| discord.client_id | string | No | Discord client ID (OAuth2) |
| discord.status_message | string | No | Bot status message |
| discord.permissions | object | No | Access control |

### behavior Section
| Key | Type | Required | Description |
|-----|------|----------|-------------|
| behavior.max_text | number | No | Max chars per message (default: 100000) |
| behavior.max_images | number | No | Max images per message (default: 5) |
| behavior.max_messages | number | No | Max messages in chain (default: 25) |
| behavior.use_plain_responses | boolean | No | Use plaintext instead of embeds |
| behavior.show_embed_color | boolean | No | Show green/orange bar |
| behavior.allow_dms | boolean | No | Allow direct messages |

### llm Section
| Key | Type | Required | Description |
|-----|------|----------|-------------|
| llm.providers | object | Yes | LLM provider configurations |
| llm.models | object | Yes | Model configurations |
| llm.fallback_models | array | No | Global fallback model list |
| llm.persona | string | No | Default persona name |
| llm.system_prompt | string | No | Global system prompt |

### Other Sections
| Key | Type | Required | Description |
|-----|------|----------|-------------|
| tools | object | No | Tool configurations |
| voice | object | No | Voice (TTS/STT) configuration |
| portal | object | No | Web portal settings |
| scheduled_tasks | object | No | Inline task definitions |

## Backward Compatibility

The config loader automatically converts legacy flat keys to the new grouped structure:

| Legacy Key | New Location |
|------------|--------------|
| bot_token | discord.bot_token |
| client_id | discord.client_id |
| status_message | discord.status_message |
| permissions | discord.permissions |
| max_text | behavior.max_text |
| max_images | behavior.max_images |
| max_messages | behavior.max_messages |
| use_plain_responses | behavior.use_plain_responses |
| show_embed_color | behavior.show_embed_color |
| allow_dms | behavior.allow_dms |
| providers | llm.providers |
| models | llm.models |
| fallback_models | llm.fallback_models |
| persona | llm.persona |
| system_prompt | llm.system_prompt |
| azure-speech | voice |

When both legacy and new keys exist, the new grouped structure takes precedence. Using legacy keys will log a deprecation warning once per key.

### Deprecation Timeline
- Legacy keys are currently supported but deprecated
- They will be removed in a future version (no timeline set yet)
- Users are encouraged to migrate to the new grouped structure

## Provider Config

```yaml
providers:
  ollama:
    base_url: "http://localhost:11434"
  openrouter:
    base_url: "https://openrouter.ai"
    api_key: "your-key"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "your-key"
```

## Model Config

```yaml
models:
  openrouter/openrouter/free:
    persona: bao
    tools: ["web_search"]
    supports_tools: true
    fallback_models:
      - "ollama/qwen3:14b"

  ollama/qwen3:14b:
    tools: ["web_search", "visuals_core"]
    think: true
    system_prompt: "You are helpful."
```

## Persona

- Stored in `bot/config/personas/<name>.(md|txt|yaml|yml)`
- Loaded by `load_persona(name)` in personas.py
- Resolution: model persona → model system_prompt → global persona → global system_prompt

## Scheduled Task

```yaml
scheduled_tasks:
  - name: daily-stock-check
    cron: "0 9 * * *"
    model: openrouter/openrouter/free
    prompt: "Check stock prices for AAPL, GOOGL"
    channel_id: 123456789
    # or user_id: 123456789
```

## Validation

- `bot/config/validator.py` validates:
  - providers exists and has entries
  - models exists and each model has valid provider
  - scheduled_tasks have required fields (name, cron, model, prompt)
  - permissions.users/roles/channels have valid structure
- Raises `ConfigValidationError` on failure