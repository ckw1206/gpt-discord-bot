## 1. Structured Formatter Implementation

- [x] 1.1 StructuredFormatter class with JSON formatting
- [x] 1.2 Environment detection (ENVIRONMENT env var)
- [x] 1.3 ISO 8601 timestamp with Z suffix
- [x] 1.4 Service and environment fields in formatter
- [x] 1.5 Optional fields support (trace_id, span_id, user_id, request_id)
- [x] 1.6 Error ext fields (error_type, error_message, stack)
- [x] 1.7 Human-readable fallback format for development

## 2. Log Handler Enhancement

- [x] 2.1 Add service and environment fields to log_handler.py
- [x] 2.2 ISO 8601 timestamp format
- [x] 2.3 trace_id and request_id propagation
- [x] 2.4 Error logging enhancement
- [x] 2.5 LOG_SAMPLING_RATE support
- [x] 2.6 WebSocket broadcast format update

## 3. Configuration Updates

- [x] 3.1 LOG_SERVICE environment variable
- [x] 3.2 bot/web/config.py service name
- [x] 3.3 Environment variables documentation
- [x] 3.4 behavior.log_level in config.yaml
- [x] 3.5 Configurable log level in llmcord.py

## 4. LogViewer Bug Fixes

- [x] 4.1 WebSocket broadcast logger field fix
- [x] 4.2 Real-time logs display fix
- [x] 4.3 Level filter end-to-end verification
- [x] 4.4 ERROR filter functionality

## 5. Domain-Specific Loggers

- [x] 5.1 Domain logger constants (discord, api, auth, llm, voice, process, db)
- [x] 5.2 "process" logger for main process
- [x] 5.3 "api" logger for HTTP requests
- [x] 5.4 "auth" logger for authentication
- [x] 5.5 "llm" logger for LLM modules
- [x] 5.6 "voice" logger for voice modules
- [x] 5.7 "db" logger for database modules
- [x] 5.8 "discord" logger for Discord events

## 6. Testing & Validation

- [x] 6.1 JSON output format (production mode)
- [x] 6.2 Human-readable format (development mode)
- [x] 6.3 Required fields in log output
- [x] 6.4 Error logging with stack trace
- [x] 6.5 Log sampling when LOG_SAMPLING_RATE < 1.0 → Verified in Section 10.3
- [x] 6.6 WebSocket log streaming
- [x] 6.7 behavior.log_level config option
- [x] 6.8 LOG_LEVEL environment variable
- [x] 6.9 Domain logger filtering in LogViewer
- [x] 6.10 LogViewer level filter

## 7. Web Portal Integration (LogViewer)

- [x] 7.1 LogViewer.tsx LogEntry interface with new fields
- [x] 7.2 Display service and environment fields
- [x] 7.3 Expandable row for trace_id, request_id, user_id
- [x] 7.4 Metadata JSON display
- [x] 7.5 Filter by service name
- [x] 7.6 Filter by environment

## 8. DEBUG Log Suppression (Performance Optimization)

- [x] 8.1 Remove DEBUG from portal.logs.levels in config
- [x] 8.2 Add filter() method to DatabaseLogHandler
- [x] 8.3 Discord.py suppression (optional)
- [x] 8.4 Application DEBUG downgrades (optional)

## 9. Time Range Filter

- [x] 9.1 Backend supports since/until query params (already implemented)
- [x] 9.2 Add timeRange state and update filtersRef
- [x] 9.3 Update fetchLogs to calculate since parameter
- [x] 9.4 Add time range dropdown UI to LogViewer

## 10. Verification (End-to-End Testing)

- [x] 10.1 DEBUG suppression: portal logs show INFO/WARNING/ERROR only
- [x] 10.2 Time range filter: works correctly with all options (fixed logger dropdown inconsistency)
- [x] 10.3 Log sampling: test when LOG_SAMPLING_RATE < 1.0
- [x] 10.4 End-to-end: all filters work together (level, logger, service, environment, time)

## 10.3 Log Sampling Verification

Verify that log sampling works correctly when LOG_SAMPLING_RATE < 1.0

- [x] 10.3.1 Test with LOG_SAMPLING_RATE=0.0 - only ERROR/CRITICAL should be logged
- [x] 10.3.2 Test with LOG_SAMPLING_RATE=0.5 - approximately 50% of INFO/WARNING sampled
- [x] 10.3.3 Verify ERROR/CRITICAL always logged regardless of sampling rate
- [x] 10.3.4 Verify sampling is random per-log (not cached per-level)
- [x] 10.3.5 Check database logs table to confirm sampling works

## 11. Uvicorn Structured Logging

Replace Uvicorn's default logging with structured JSON output to match the application logging.

### 11.1 Uvicorn Log Configuration

- [x] 11.1.1 Configure Uvicorn to use our StructuredFormatter
- [x] 11.1.2 Override uvicorn.default logger
- [x] 11.1.3 Override uvicorn.access logger for HTTP request logs
- [x] 11.1.4 Override uvicorn.asgi logger

### 11.2 Python Warnings Suppression

- [x] 11.2.1 Configure logging to suppress DeprecationWarning
- [x] 11.2.2 Set warnings.filterwarnings in main entry point
- [x] 11.2.3 Test that datetime.utcnow() warning is suppressed

### 11.3 Verification

- [x] 11.3.1 Verify Uvicorn startup logs are JSON structured
- [x] 11.3.2 Verify HTTP access logs are JSON structured
- [x] 11.3.3 Verify no DeprecationWarning in output
- [x] 11.3.4 Verify development mode still shows readable format

## 12. Documentation

- [x] 12.1 Update CONTRIBUTING.md with logging standards → Created in docs/logging.md (more appropriate location)
- [x] 12.2 Document new environment variables → docs/logging.md covers ENVIRONMENT, LOG_LEVEL, LOG_SERVICE, LOG_SAMPLING_RATE
- [x] 12.3 Document behavior.log_level config option → docs/logging.md covers config.yaml settings
- [x] 12.4 Document domain logger usage → docs/logging.md covers all domain loggers with examples

## Summary

| Section | Status | Notes |
|---------|--------|-------|
| 1. Structured Formatter | ✅ Complete | |
| 2. Log Handler Enhancement | ✅ Complete | |
| 3. Configuration Updates | ✅ Complete | |
| 4. LogViewer Bug Fixes | ✅ Complete | |
| 5. Domain-Specific Loggers | ✅ Complete | |
| 6. Testing & Validation | ✅ Complete | |
| 7. Web Portal Integration | ✅ Complete | |
| 8. DEBUG Log Suppression | ✅ Complete | |
| 9. Time Range Filter | ✅ Complete | frontend + backend |
| 10. Verification | ✅ Complete | |
| 11. Uvicorn Structured Logging | ✅ Complete | |
| 12. Documentation | ✅ Complete | docs/logging.md |

## Implementation Status

**Ready**: Structured logging system fully implemented and documented (JSON in prod, readable in dev, configurable log level, DEBUG suppressed in portal, domain loggers, time range filter, consistent logger dropdown, Uvicorn structured logging, Python warnings suppression, documentation in docs/logging.md)

**Complete**: All tasks including documentation (in docs/logging.md)

