## Context

Multiple test failures were discovered across the test suite that need to be fixed:

### Issue Categories:

1. **Voice Config Env Var Fallback** (Related to our changes)
   - `test_voice_config.py`: 5 failures in `TestVoiceConfigEnvOverrides`
   - `test_voice_config_standalone.py`: 0 failures (already passing with our fix)

2. **Test Infrastructure Issues** (Pre-existing)
   - `test_web_log_handler.py`: 14 failures - Missing `sqlalchemy` dependency
   - `test_bot_integration.py`: 3 failures - Missing `@pytest_asyncio.fixture`
   - `test_frontend_integration.py`: 7 failures - Frontend file checks
   - `test_httpx_discord.py`: 6 failures - Network tests need mocking
   - `test_web_file_views.py`: 3 failures - Task runner path resolution

**Current State:**
- Voice config: implemented `_get_with_fallback()` function (needs verification)
- Test infrastructure: Multiple pre-existing issues

**Constraints:**
- Must maintain backward compatibility
- Should not change behavior when config values are provided

## Goals / Non-Goals

**Goals:**
- Fix voice config env var fallback (already implemented)
- Fix test infrastructure issues across all failing test files
- Get all tests passing or properly marked as skip/xfail

**Non-Goals:**
- No new capabilities - just fixing implementation and test issues
- No changes to existing working functionality

## Decisions

### Decision 1: Where to implement env var fallback

**Chosen:** Add fallback in `bot/voice/config.py` `_get_with_fallback()` method

**Rationale:**
- Keeps fallback logic close to where config is consumed
- Allows voice config to work independently of the main config loader normalization
- Maintains separation of concerns

**Alternative Considered:** Add fallback in config loader normalization layer
- Rejected: Would apply globally, but voice config has its own specific env var names

### Decision 2: How to fix test infrastructure

**Chosen:** Case-by-case approach based on root cause

| Test File | Root Cause | Fix Approach |
|-----------|------------|--------------|
| test_web_log_handler.py | Missing sqlalchemy | Add try/except or mock database imports |
| test_bot_integration.py | Missing async fixtures | Add proper `@pytest_asyncio.fixture` decorators |
| test_frontend_integration.py | File existence checks | Update to check actual dist folder or skip |
| test_httpx_discord.py | Network tests | Mark as integration tests or add proper mocks |
| test_web_file_views.py | Task runner path | Fix path resolution logic |

## Risks / Trade-offs

- **Risk**: Test fixes might mask real issues
  - **Mitigation**: Ensure tests properly verify behavior, not just pass

- **Risk**: Web build fails due to TypeScript errors
  - **Mitigation**: Already verified working

- **Risk**: Database-related tests require complex mocking
  - **Mitigation**: Use `@pytest.mark.skip` for integration tests that need DB

## Migration Plan

1. Verify voice config tests pass (already implemented)
2. Run full test suite to get exact failure count
3. Fix each test file one by one:
   - test_web_log_handler.py
   - test_bot_integration.py
   - test_frontend_integration.py
   - test_httpx_discord.py
   - test_web_file_views.py
4. Run full test suite to confirm all pass

## Open Questions

None - the implementation approach is straightforward.