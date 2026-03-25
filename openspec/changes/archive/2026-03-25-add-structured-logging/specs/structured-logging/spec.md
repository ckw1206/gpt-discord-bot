## ADDED Requirements

### Requirement: Structured JSON Logging
The system SHALL output logs in structured JSON format in production environments, containing all required fields defined in the logging-guide skill.

#### Scenario: Production JSON logging
- **WHEN** ENVIRONMENT is set to "production"
- **THEN** logs are output as valid JSON with required fields

#### Scenario: Development readable logging
- **WHEN** ENVIRONMENT is not set or set to "development"
- **THEN** logs are output in human-readable format

### Requirement: Required Log Fields
All log entries SHALL include the following required fields:
- timestamp (ISO 8601 format with Z suffix)
- level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- message (log content)
- service (service identifier, default: "discord-bot")
- environment (from ENVIRONMENT env var)

#### Scenario: Log entry contains all required fields
- **WHEN** any log statement is executed
- **THEN** the output contains timestamp, level, message, service, and environment fields

### Requirement: Recommended Log Fields
Log entries SHALL include recommended fields when context is available:
- trace_id: for request tracing
- span_id: for distributed tracing
- user_id: when user context is available
- request_id: for HTTP request correlation

#### Scenario: Log with user context
- **WHEN** logging with user context available
- **THEN** the log includes user_id field

#### Scenario: Log with request context
- **WHEN** logging during HTTP request processing
- **THEN** the log includes request_id field

### Requirement: Error Log Context
Error-level logs SHALL include additional context for debugging:
- error_type: exception class name
- error_message: exception message
- stack: full stack trace (when available)

#### Scenario: Error logging with exception
- **WHEN** logging an exception at ERROR or CRITICAL level
- **THEN** the log includes error_type, error_message, and stack fields

### Requirement: Log Sampling
The system SHALL support log sampling in production environments to prevent log bloat.

#### Scenario: High-traffic sampling
- **WHEN** LOG_SAMPLING_RATE is set less than 1.0 in production
- **THEN** only the configured percentage of logs are written

#### Scenario: Default full logging
- **WHEN** LOG_SAMPLING_RATE is not configured
- **THEN** all logs are written (100% sampling)

### Requirement: Environment-Aware Log Levels
The system SHALL use different log levels based on environment:
- Development: DEBUG level enabled
- Production: INFO level (DEBUG disabled)

#### Scenario: Development debug logging
- **WHEN** ENVIRONMENT is "development"
- **THEN** DEBUG level logs are visible

#### Scenario: Production info logging
- **WHEN** ENVIRONMENT is "production"
- **THEN** DEBUG level logs are suppressed

### Requirement: Web Portal Log Viewer Integration
The web portal's LogViewer component SHALL display structured log fields from the database.

#### Scenario: LogViewer displays structured fields
- **WHEN** logs are retrieved from `/api/logs` endpoint
- **THEN** LogViewer displays service, environment, and metadata fields

#### Scenario: LogViewer shows metadata expansion
- **WHEN** log entry contains trace_id, request_id, or user_id
- **THEN** these fields are accessible via expandable row or metadata display

#### Scenario: LogViewer field mapping
- **WHEN** viewing logs in web portal
- **THEN** the logger column maps to event_type, metadata column maps to extra_data JSON

**Field Mapping Table**:
| LogViewer Field | API Response | Database Column | Notes |
|-----------------|--------------|-----------------|-------|
| logger | event_type | event_type | Service/logger identifier |
| metadata | metadata | extra_data (JSON) | Contains trace_id, request_id, user_id |

### Requirement: Configurable Log Level
The system SHALL support configurable log level via config.yaml or environment variable, with the following precedence order (highest to lowest):

1. `LOG_LEVEL` environment variable (highest priority)
2. `behavior.log_level` in config.yaml
3. Environment-based default (DEBUG for development, INFO for production)

#### Scenario: LOG_LEVEL environment variable override
- **WHEN** LOG_LEVEL environment variable is set to "WARNING"
- **THEN** logs below WARNING level are suppressed regardless of config.yaml

#### Scenario: behavior.log_level in config.yaml
- **WHEN** config.yaml has `behavior.log_level: ERROR`
- **THEN** only ERROR and CRITICAL logs are output

#### Scenario: Fallback to environment-based default
- **WHEN** neither LOG_LEVEL env var nor behavior.log_level is configured
- **THEN** DEBUG level is used in development, INFO level in production

#### Scenario: Invalid log level configuration
- **WHEN** an invalid log level is specified (e.g., "INVALID")
- **THEN** the system SHALL fall back to DEBUG level and log a warning

### Requirement: Domain-Specific Loggers
The system SHALL use domain-specific named loggers for key application areas to enable filtering and better log organization.

The following domain loggers MUST be available:
- `discord`: Discord bot events, commands, message handling
- `api`: HTTP API requests and responses
- `auth`: Authentication events (login, logout, token operations)
- `llm`: LLM interactions (prompts, responses, tool calls)
- `voice`: Voice operations (STT, TTS)
- `process`: Main process, scheduler, startup/shutdown
- `db`: Database operations

#### Scenario: Domain logger usage
- **WHEN** logging in the API layer
- **THEN** the logger name in the log entry SHALL be "api"

#### Scenario: LogViewer filtering by domain
- **WHEN** user selects "api" in the logger filter dropdown
- **THEN** only logs from the api logger are displayed

#### Scenario: All domain loggers produce structured output
- **WHEN** any domain logger is used
- **THEN** the log output includes all required structured fields (timestamp, level, message, service, environment)

### Requirement: LogViewer Log Level Filter
The system SHALL support filtering logs by level in the LogViewer web portal component.

#### Scenario: Level filter shows only selected level
- **WHEN** user selects "ERROR" in the LogViewer level dropdown
- **THEN** only logs with level ERROR or CRITICAL are displayed

#### Scenario: Level filter shows all levels
- **WHEN** user selects "All Levels" in the LogViewer level dropdown
- **THEN** all logs matching other filters are displayed

#### Scenario: Real-time logs display correctly via WebSocket
- **WHEN** new logs arrive via WebSocket connection
- **THEN** the logs display in LogViewer with all fields properly populated
- **AND** the logger field is present and matches the API response format

#### Scenario: WebSocket logger field compatibility
- **WHEN** logs are broadcast via WebSocket
- **THEN** the broadcast data SHALL include a `logger` field (mapping from `event_type`)
- **AND** LogViewer can display the logger name correctly

### Requirement: LogViewer Time Range Filter
The system SHALL support filtering logs by time range in the LogViewer web portal component with predefined options.

#### Scenario: Time range filter options
- **WHEN** user opens the time range dropdown in LogViewer
- **THEN** the following options are available:
  - All Time (default)
  - Last 1 Hour
  - Last 4 Hours
  - Last 8 Hours
  - Last 1 Day
  - Last 7 Days

#### Scenario: Time range filter applies to API request
- **WHEN** user selects a time range (e.g., "Last 1 Hour")
- **THEN** the API request to `/api/logs` includes a `since` parameter with the calculated ISO 8601 timestamp
- **AND** only logs within the selected time range are returned

#### Scenario: Time range works with other filters
- **WHEN** user selects a time range AND other filters (level, logger, service, environment)
- **THEN** all filters are combined with AND logic
- **AND** logs must match both the time range and the other filter criteria

#### Scenario: "All Time" removes time filter
- **WHEN** user selects "All Time" in the dropdown
- **THEN** no `since` parameter is sent to the API
- **AND** all logs matching other filters are returned

#### Scenario: Backend already supports time filtering
- **WHEN** API receives `/api/logs?since=2024-01-15T10:00:00Z`
- **THEN** the backend filters logs where timestamp >= since parameter
- **AND** the SQL filtering is already implemented in logs.py

### Requirement: DEBUG Log Suppression for Portal
The system SHALL suppress DEBUG level logs from being stored in the database portal to prevent log bloat while preserving console output.

#### Scenario: DEBUG not stored in portal
- **WHEN** DEBUG level log is generated
- **THEN** it is NOT stored in the database via DatabaseLogHandler
- **AND** the log still appears in console output (if console handler level permits)

#### Scenario: Configurable portal log levels
- **WHEN** config.yaml has `portal.logs.levels` configured
- **THEN** only those levels are stored in the database
- **AND** levels not in the list (e.g., DEBUG) are filtered out

#### Scenario: Level filter shows only enabled levels
- **WHEN** LogViewer fetches available levels from `/api/logs/levels`
- **THEN** only levels enabled in config are shown in the dropdown
- **AND** disabled levels return empty results (not all logs)