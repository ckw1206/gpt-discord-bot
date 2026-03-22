## MODIFIED Requirements

### Requirement: Azure-speech config fields defined
The azure-speech config SHALL support these fields:

#### Scenario: Key field from config
- **WHEN** `key` field is provided in azure-speech config
- **THEN** used for Azure Speech API authentication

#### Scenario: Region field from config
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
- **THEN** use `AZURE_SPEECH_KEY` value as the key

#### Scenario: Region from AZURE_SPEECH_REGION env var
- **WHEN** `region` field in config is empty AND `AZURE_SPEECH_REGION` environment variable is set
- **THEN** use `AZURE_SPEECH_REGION` value as the region

#### Scenario: Voice from AZURE_SPEECH_VOICE env var
- **WHEN** `default_voice` field in config is empty AND `AZURE_SPEECH_VOICE` environment variable is set
- **THEN** use `AZURE_SPEECH_VOICE` value as the default voice

#### Scenario: Style from AZURE_SPEECH_STYLE env var
- **WHEN** `default_style` field in config is empty AND `AZURE_SPEECH_STYLE` environment variable is set
- **THEN** use `AZURE_SPEECH_STYLE` value as the default style