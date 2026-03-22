# voice-config Specification

## Purpose
TBD - created by archiving change azure-voice-integration. Update Purpose after archive.
## Requirements
### Requirement: Azure-speech config section in config.yaml
The system SHALL support an `azure-speech` section in config.yaml with required and optional fields.

#### Scenario: Valid azure-speech config
- **WHEN** config.yaml contains:
  ```yaml
  azure-speech:
    key: "your-azure-speech-key"
    region: "eastus"
  ```
- **THEN** the config is considered valid

#### Scenario: Missing required fields
- **WHEN** azure-speech section exists but missing `key` or `region`
- **THEN** config validation fails with appropriate error message

#### Scenario: Azure-speech section optional
- **WHEN** azure-speech section is not present in config.yaml
- **THEN** voice features are disabled, bot works without them

### Requirement: Azure-speech config fields defined
The azure-speech config SHALL support these fields:

#### Scenario: Key field
- **WHEN** `key` field is provided in azure-speech config
- **THEN** used for Azure Speech API authentication

#### Scenario: Region field
- **WHEN** `region` field is provided (e.g., "eastus", "westeurope")
- **THEN** used to determine Azure Speech endpoint region

#### Scenario: Optional endpoint field
- **WHEN** `endpoint` field is provided in azure-speech config
- **THEN** used as custom Azure Speech endpoint (optional override)

#### Scenario: Default voice field
- **WHEN** `default_voice` field is provided (e.g., "en-US-JennyNeural")
- **THEN** used as the default voice for TTS when not specified

#### Scenario: Default style field
- **WHEN** `default_style` field is provided (e.g., "cheerful", "sad")
- **THEN** used as the default speaking style for TTS

### Requirement: Environment variable fallback
The azure-speech config SHALL fall back to environment variables when config values are empty.

#### Scenario: Key from AZURE_SPEECH_KEY env var
- **WHEN** `key` field in config is empty AND `AZURE_SPEECH_KEY` environment variable is set
- **THEN** use the environment variable value for authentication

#### Scenario: Region from AZURE_SPEECH_REGION env var
- **WHEN** `region` field in config is empty AND `AZURE_SPEECH_REGION` environment variable is set
- **THEN** use the environment variable value for endpoint region

#### Scenario: Endpoint from AZURE_SPEECH_ENDPOINT env var
- **WHEN** `endpoint` field in config is empty AND `AZURE_SPEECH_ENDPOINT` environment variable is set
- **THEN** use the environment variable value for custom endpoint

#### Scenario: Voice from AZURE_SPEECH_VOICE env var
- **WHEN** `default_voice` field in config is empty AND `AZURE_SPEECH_VOICE` environment variable is set
- **THEN** use the environment variable value for default voice

#### Scenario: Style from AZURE_SPEECH_STYLE env var
- **WHEN** `default_style` field in config is empty AND `AZURE_SPEECH_STYLE` environment variable is set
- **THEN** use the environment variable value for default style

#### Scenario: Env vars override empty config
- **WHEN** config has empty fields but environment variables are set
- **THEN** use environment variable values
- **AND** config values take precedence when present

