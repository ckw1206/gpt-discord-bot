## Why

During development and testing, multiple test failures were discovered across the test suite. These need to be fixed to ensure the project is in a working state:

1. **Voice config env var fallback** - `test_voice_config.py` and `test_voice_config_standalone.py` fail because the voice config module doesn't support automatic environment variable fallback when config values are empty.

2. **Test infrastructure issues** - Various tests fail due to:
   - Missing `sqlalchemy` dependency (DatabaseLogHandler tests)
   - Missing `pytest-asyncio` fixtures (async tests)
   - Pre-existing test issues (network tests, frontend file checks, task runner)

## What Changes

### Voice Config
- Add automatic env var fallback to `bot/voice/config.py`:
  - `AZURE_SPEECH_KEY` → `key` field
  - `AZURE_SPEECH_REGION` → `region` field
  - `AZURE_SPEECH_VOICE` → `default_voice` field
  - `AZURE_SPEECH_STYLE` → `default_style` field
- Verify `npm run build` works in web directory

### Test Infrastructure Fixes
- Fix `test_web_log_handler.py` - install missing dependencies or mock database imports
- Fix `test_bot_integration.py` - add proper async fixtures
- Fix `test_frontend_integration.py` - update file existence checks
- Fix `test_httpx_discord.py` - mark as integration tests or skip in CI
- Fix `test_web_file_views.py` - fix task runner path resolution

## Capabilities

### New Capabilities
None - this is a bug fix and test infrastructure improvement task.

### Modified Capabilities
None - existing functionality is unchanged.

## Impact

- **Modified**: `bot/voice/config.py` - add env var fallback logic
- **Modified**: `bot/test/` - fix test infrastructure issues
- **Modified**: `web/` - verify build works with ConfigEditor changes
- **Tests Fixed**: ~30 tests across multiple test files