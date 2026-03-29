## MODIFIED Requirements

### Requirement: Config schema defines required structure
The config SHALL have these top-level keys:

**New grouped structure:**
- `discord`: Discord bot identity and permissions
  - `bot_token`: Discord bot token (required)
  - `client_id`: Discord client ID (optional)
  - `status_message`: Bot status message (optional)
  - `permissions`: Access control configuration (optional)
- `behavior`: Bot behavior settings
  - `max_text`: Max chars per message (default: 100000)
  - `max_images`: Max images per message (default: 5)
  - `max_messages`: Max messages in chain (default: 25)
  - `use_plain_responses`: Use plaintext instead of embeds (default: false)
  - `show_embed_color`: Show green/orange bar (default: true)
  - `allow_dms`: Allow direct messages (default: true)
- `llm`: LLM configuration
  - `providers`: LLM provider configurations (required)
  - `models`: Model configurations (required)
  - `persona`: Default persona name (optional)
  - `system_prompt`: Global system prompt (optional)
  - `fallback_models`: Global fallback model list (optional)
- `tools`: Tool configurations (optional)
- `voice`: Voice/TTS configuration (optional, renamed from azure-speech)
- `portal`: Web portal configuration (optional)
- `scheduled_tasks`: Inline task definitions (optional)

**Legacy flat keys (deprecated, still supported):**
- `bot_token`, `client_id`, `status_message`, `permissions` (→ discord.*)
- `max_text`, `max_images`, `max_messages`, `use_plain_responses`, `show_embed_color`, `allow_dms` (→ behavior.*)
- `providers`, `models`, `persona`, `system_prompt`, `fallback_models` (→ llm.*)
- `azure-speech` (→ voice)

#### Scenario: New grouped config structure
- **WHEN** AI reads config.yaml with grouped sections
- **THEN** it can parse: discord, behavior, llm, tools, voice, portal, scheduled_tasks

#### Scenario: Legacy flat config structure (backward compat)
- **WHEN** AI reads config.yaml with flat top-level keys
- **THEN** normalization layer converts them to grouped structure with deprecation warning

### Requirement: Config supports both new and legacy keys
The system SHALL accept configuration in both grouped and flat formats, with grouped format preferred.

#### Scenario: Mixed format
- **WHEN** config has some keys in grouped format and others in legacy format
- **THEN** normalization layer merges both, with grouped format taking precedence for conflicts

### Requirement: Deprecation warnings for legacy keys
The system SHALL log a warning when legacy flat keys are used, indicating the new preferred structure.

#### Scenario: Legacy key warning
- **WHEN** config loads with legacy keys (bot_token, providers, etc.)
- **THEN** system logs deprecation warning once per key