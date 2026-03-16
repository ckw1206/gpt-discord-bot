# Azure Speech Configuration

This guide explains how to enable voice features (TTS and STT) using Azure Speech Services.

## Prerequisites

- An Azure subscription
- A Speech resource created in [Azure Portal](https://portal.azure.com/#create/Microsoft.CognitiveServicesSpeechServices)

## Setup

1. **Get your Azure Speech credentials**
   - Go to [Azure Portal](https://portal.azure.com) → Your Speech resource
   - Copy the **Key** (either Key1 or Key2)
   - Note your **Location/Region** (e.g., `eastus`, `westeurope`, `japaneast`)

2. **Update config.yaml**
   ```yaml
   azure-speech:
     key: "your-azure-speech-key"
     region: "eastus"
   ```

## Configuration Options

| Field | Required | Description |
|-------|----------|-------------|
| `key` | Yes | Your Azure Speech API key |
| `region` | Yes | Your Azure region (e.g., eastus, westeurope) |
| `endpoint` | No | Custom endpoint (rarely needed) |
| `default_voice` | No | Default voice for TTS (e.g., `en-US-JennyNeural`) |
| `default_style` | No | Default speaking style (e.g., `cheerful`, `sad`, `neutral`) |

## Available Voices

You can find all available Azure voices on the [Microsoft documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/text-to-speech).

Popular English voices:
- `en-US-JennyNeural` - Female, conversational
- `en-US-GuyNeural` - Male, professional
- `en-GB-SoniaNeural` - Female, British
- `en-AU-NatashaNeural` - Female, Australian

**Chinese voices (use when speaking Chinese/Taiwanese):**
- `zh-CN-XiaoxiaoNeural` - Female, Mandarin Chinese (recommended)
- `zh-CN-YunxiNeural` - Male, Mandarin Chinese
- `zh-TW-HsiaoChenNeural` - Female, Taiwanese Mandarin
- `zh-TW-YunJheNeural` - Male, Taiwanese Mandarin
- `zh-HK-HiuMaanNeural` - Female, Hong Kong Cantonese

> **Tip:** The bot automatically detects the voice locale (e.g., `zh-CN` from `zh-CN-XiaoxiaoNeural`) and sets the correct language for TTS. Use `zh-CN` voices for Simplified Chinese, `zh-TW` for Traditional Chinese (Taiwan), or `zh-HK` for Cantonese.

## Speaking Styles

Available styles vary by voice, but common ones include:
- `cheerful` - Happy, positive
- `sad` - Depressed, somber
- `angry` - Frustrated, upset
- `neutral` - Default, neutral tone
- `excited` - Enthusiastic
- `friendly` - Warm, approachable
- `whispering` - Quiet, soft

## Voice Commands

Once configured, the following commands are available:

| Command | Description |
|---------|-------------|
| `/join` | Join your current voice channel |
| `/leave` | Leave the current voice channel |
| `/speak <text>` | Make the bot speak the text in voice channel |

## Voice Messages

When Azure Speech is configured, the bot can also:
- **Transcribe voice messages** sent in text channels or DMs
- Process audio attachments automatically

The STT service:
- Automatically uses Chinese language (zh-TW) for better recognition of Chinese speech
- Converts Discord voice messages (OGG format) to the required 16kHz mono WAV format
- Returns Chinese characters instead of romanization (e.g., "你好" not "NI hao")

### Voice Message Requirements

For best transcription results:
- Use clear, natural speech
- Avoid background noise
- Chinese voice messages work best with zh-TW locale voices

## Testing

Test your TTS configuration by:
1. Joining a voice channel
2. Running `/speak Hello world`

The bot should speak the text aloud in the voice channel.

Test your STT configuration by:
1. Sending a voice message in a text channel or DM
2. The bot should transcribe and respond to the content

## Enabling/Disabling Voice Features

Voice features (TTS and STT) are automatically enabled when valid credentials are provided. To toggle on/off:

| Method | How to Enable | How to Disable |
|--------|---------------|----------------|
| **Set credentials** | `key` and `region` both set | Leave both empty or comment out |
| **Use environment variables** | Set `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` env vars | Unset the env vars |
| **Config file** | `key: "your-key"`<br>`region: "eastus"` | `key: ""`<br>`region: ""` |

The bot checks `is_configured` which returns `True` only when both `key` and `region` are present and not empty.

## Troubleshooting

- **"TTS is not configured"**: Ensure `key` and `region` are set in config.yaml
- **Authentication errors**: Verify your Azure key is correct
- **Region mismatch**: Ensure the region matches your Speech resource's location
- **Bot can't join voice**: Check that the bot has "Connect" and "Speak" permissions in the voice channel
- **STT returns wrong language**: Ensure the voice message is clear and use zh-TW locale voices for Chinese
- **STT returns romanization**: This happens with auto-detect; the bot now explicitly sets zh-TW for Chinese speech