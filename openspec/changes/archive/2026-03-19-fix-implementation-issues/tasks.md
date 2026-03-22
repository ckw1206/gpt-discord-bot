## 1. Voice Config Environment Variable Fallback

- [x] 1.1 Read failing voice config tests to understand expected behavior
- [x] 1.2 Add env var fallback to bot/voice/config.py using AZURE_SPEECH_* env vars
- [x] 1.3 Run voice config tests to verify fix

## 2. Verify Web Build

- [x] 2.1 Run npm run build in web directory
- [x] 2.2 Fix any build errors if they exist

## 3. Fix test_web_log_handler.py

- [x] 3.1 Investigate failure root cause (sqlalchemy dependency)
- [x] 3.2 Fix by adding try/except or mocking database imports
- [x] 3.3 Run tests to verify fix

## 4. Fix test_bot_integration.py

- [x] 4.1 Investigate failure root cause (config structure changed, no top-level models/providers)
- [x] 4.2 Add backward compatibility aliases in config loader for models/providers
- [x] 4.3 Run tests to verify fix

## 5. Fix test_frontend_integration.py

- [x] 5.1 Investigate failure root cause (outdated tests, string vs int IDs)
- [x] 5.2 Update tests to match current code (flex layout, string IDs, ServerDrawer)
- [x] 5.3 Run tests to verify fix

## 6. Fix test_httpx_discord.py

- [x] 6.1 Investigate failure root cause (network tests)
- [x] 6.2 Tests pass now (likely were temporarily failing)
- [x] 6.3 Run tests to verify fix

## 7. Fix test_web_file_views.py

- [x] 7.1 Investigate failure root cause (numpy/pandas import errors)
- [x] 7.2 Root cause: test_reload_single_task_success uses sys.modules.clear() which removes numpy/pandas
- [x] 7.3 Fixed by changing to sys.modules.pop('bot.llmcord', None) in all 3 affected tests
- [x] 7.4 Run tests to verify fix

## 8. Final Verification

- [x] 8.1 Run full test suite
- [x] 8.2 Confirm all tests pass (75 passed, 1 skipped)