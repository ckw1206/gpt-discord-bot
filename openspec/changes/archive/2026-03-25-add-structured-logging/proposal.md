## Why

The current logging implementation uses simple text format without structured fields, making log analysis, debugging, and observability difficult. Logs lack essential fields like service name, environment, request IDs, and proper error context. Implementing structured logging will improve debugging, enable better log aggregation, and align with the logging-guide skill best practices.

Additionally, the current system lacks:
- A centralized log level configuration in config.yaml (currently hardcoded based on ENVIRONMENT)
- Domain-based log separation (all logs go through a single logger, making filtering difficult)

## What Changes

- Replace basic `logging.basicConfig` with custom `StructuredFormatter` class
- Output logs in JSON format for production, human-readable for development
- Add required fields: timestamp (ISO 8601), level, message, service, environment
- Add recommended fields: trace_id, span_id, user_id, request_id
- Enhance error logging with error_type, error_message, stack trace
- Add log sampling for high-traffic endpoints in production
- Update `DatabaseLogHandler` to include structured fields
- Add `behavior.log_level` configuration in config.yaml with environment variable fallback
- Implement domain-specific loggers (discord, api, auth, llm, voice, process, db) for better log categorization
- **Fix LogViewer LogLevel filter**: WebSocket broadcast sends `event_type`, LogViewer expects `logger` field

## Capabilities

### New Capabilities

- `structured-logging`: Unified logging format with all required and recommended fields from logging-guide skill

### Modified Capabilities

- None - this is a new capability not modifying existing spec requirements

## Impact

- **Files Modified**: `llmcord.py`, `bot/web/log_handler.py`, `bot/web/config.py`, `bot/web/routes/logs.py`, `web/src/components/LogViewer.tsx`, `config.yaml`, `config-example.yaml`
- **Dependencies**: None (using Python standard library)
- **Configuration**: 
  - New environment variables: `ENVIRONMENT`, `LOG_SERVICE`, `LOG_SAMPLING_RATE`, `LOG_LEVEL`
  - New config option: `behavior.log_level` in config.yaml (env var LOG_LEVEL as fallback)
- **Web Portal Impact**: 
  - LogViewer.tsx needs updates to display new structured fields (service, environment, trace_id, request_id, user_id)
  - **Bug Fix**: WebSocket broadcast format fix - add `logger` field to match LogViewer expectations
- **New Loggers**: Domain-specific loggers for filtering: discord, api, auth, llm, voice, process, db
- **DEBUG Log Suppression**: Remove DEBUG from portal.logs.levels to prevent database I/O flood from SQLAlchemy and third-party library logs
  - Implemented via filter() method in DatabaseLogHandler
  - Core fix complete; optional enhancements pending (discord.py suppression)