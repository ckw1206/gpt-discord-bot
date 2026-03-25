## Context

The current logging implementation uses basic text format via `logging.basicConfig` in `llmcord.py`. The log handler in `bot/web/log_handler.py` writes to both console and database but lacks structured fields.

Current state:
- Simple format: `%(asctime)s %(levelname)s: %(message)s`
- No service name or environment tracking
- No request/trace IDs for correlation
- Basic error logging without context

Using the logging-guide skill, we need to implement structured JSON logging with all required and recommended fields.

## Goals / Non-Goals

**Goals:**
- Implement structured JSON logging for production environment
- Add all required fields: timestamp (ISO 8601), level, message, service, environment
- Add recommended fields: trace_id, span_id, user_id, request_id
- Enhance error logging with error_type, error_message, stack
- Add log sampling for high-traffic endpoints
- Maintain human-readable format for development

**Non-Goals:**
- Integration with external log aggregation services (Datadog, Splunk, etc.)
- OpenTelemetry tracing implementation
- Log retention policies (already exists)

## Decisions

### 1. Structured Formatter Class
**Decision**: Create a custom `StructuredFormatter` class in `llmcord.py`

**Rationale**: Python's built-in formatters don't support JSON output. A custom class allows environment-aware formatting (JSON in prod, readable in dev).

**Alternative Considered**: Use `python-json-logger` library
- Rejected: Adds external dependency; custom class uses only stdlib

### 2. JSON vs Human-Readable
**Decision**: JSON format for production, readable format for development

**Rationale**: 
- Production: JSON enables log aggregation and parsing
- Development: Readable format aids debugging

**Environment Detection**: `ENVIRONMENT` environment variable (defaults to "development")

### 3. Field Implementation
**Decision**: Add fields via both formatter and log handler

- **Formatter** (`llmcord.py`): Core fields (timestamp, level, message, service, environment)
- **Handler** (`log_handler.py`): Context fields (request_id, user_id) and database-specific fields

**Rationale**: Separates concerns - formatter handles output, handler adds context-specific data

### 4. Error Context Enhancement
**Decision**: Add error_type, error_message, stack to error logs

**Rationale**: Following logging-guide skill's error logging requirements for actionable debugging

### 5. Log Sampling
**Decision**: Add sampling rate configuration for production

**Rationale**: Prevents log bloat in high-traffic scenarios; follows skill's performance considerations

**Configuration**: `LOG_SAMPLING_RATE` env var (0.0-1.0, default 1.0)

### 6. Web Portal Integration (LogViewer)
**Decision**: Store structured fields in `extra_data` (JSON), expose via API, display in LogViewer

**Rationale**: 
- LogViewer.tsx already uses `logger` field (maps to `event_type` in DB)
- New fields (service, environment, trace_id, request_id, user_id) will be stored in `extra_data` JSON
- API returns `metadata` field which maps to `extra_data` - already handled by LogViewer

**Implementation**:
- Structured fields from logging will be stored in `EventLog.extra_data` JSON column
- API in `logs.py` already passes through `metadata` field
- LogViewer.tsx needs minor updates to display specific fields from metadata

**Field Mapping**:

### 7. Configurable Log Level (behavior.log_level)
**Decision**: Add `behavior.log_level` option in config.yaml with LOG_LEVEL environment variable as fallback

**Rationale**:
- Current implementation hardcodes log level based on ENVIRONMENT variable
- Users want explicit control over log level independent of environment
- Environment variable provides flexibility for containerized deployments

**Configuration Precedence** (highest to lowest):
1. `LOG_LEVEL` environment variable (highest priority, for docker/k8s deployments)
2. `behavior.log_level` in config.yaml (explicit configuration)
3. Default: DEBUG in development, INFO in production (fallback behavior)

**Implementation**:
```yaml
# config.yaml
behavior:
  log_level: DEBUG  # Optional: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```python
# llmcord.py - Log level resolution
def get_log_level():
    # 1. Check LOG_LEVEL env var first
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        return getattr(logging, env_level.upper(), logging.DEBUG)
    
    # 2. Check config.yaml behavior.log_level
    config = get_config()
    config_level = config.get("behavior", {}).get("log_level")
    if config_level:
        return getattr(logging, config_level.upper(), logging.DEBUG)
    
    # 3. Fallback to environment-based default
    environment = os.environ.get("ENVIRONMENT", "development")
    return logging.DEBUG if environment == "development" else logging.INFO
```

### 8. Domain-Specific Loggers
**Decision**: Implement separate logger instances for different application domains

**Rationale**:
- Current: All modules use `logging.getLogger(__name__)` which gives full module path
- Problem: Hard to filter logs by domain (e.g., "show only API logs")
- Solution: Use named loggers for key domains that appear in LogViewer

**Logger Names**:
| Domain | Logger Name | Purpose |
|--------|-------------|---------|
| Discord Bot | `discord` | Bot events, commands, messages |
| HTTP API | `api` | Web API requests, responses |
| Authentication | `auth` | Login, logout, token operations |
| LLM | `llm` | Model prompts, responses, tool calls |
| Voice | `voice` | STT/TTS operations |
| Process | `process` | Main process, scheduler, startup |
| Database | `db` | Database operations |

**Implementation**:
- Each domain module uses its named logger (e.g., `logging.getLogger("api")`)
- DatabaseLogHandler captures logger name in `event_type` field
- LogViewer can filter by these domain loggers using the existing logger filter dropdown

**Example Usage**:
```python
# Instead of: logger = logging.getLogger(__name__)
# Use domain-specific loggers:
api_logger = logging.getLogger("api")
api_logger.info("Request received", extra={"request_id": request_id})

discord_logger = logging.getLogger("discord")
discord_logger.info("User message received", extra={"user_id": user_id})
```
| UI Field (LogViewer) | API Field (logs.py) | DB Field (models.py) |
|---------------------|---------------------|---------------------|
| logger | event_type | event_type |
| metadata | metadata | extra_data (JSON) |

**New fields to display in LogViewer**:
- service (from LOG_SERVICE env var)
- environment (from ENVIRONMENT env var)
- trace_id, request_id, user_id (from extra_data)

| Risk | Impact | Mitigation |
|------|--------|------------|
| JSON format breaks existing log parsing | Medium | Environment-aware: only prod uses JSON |
| Performance overhead of JSON serialization | Low | Minimal; Python's json module is fast |
| Missing context in some log calls | Medium | Use logging.Logger's extra parameter consistently |
| Backward compatibility | Low | Development env keeps readable format |

## 9. LogViewer Bug Fixes

### Issue: WebSocket Field Mismatch
**Problem**: The WebSocket broadcast in `log_handler.py` sends `event_type` field, but LogViewer.tsx expects `logger` field.

**Impact**: Real-time logs via WebSocket show empty/undefined logger name in LogViewer.

**Location**: `bot/web/log_handler.py` line ~178

**Current (buggy):**
```python
broadcast_data = {
    "timestamp": log_data["timestamp"],
    "level": log_data["level"],
    "event_type": log_data["event_type"],  # LogViewer expects "logger"
    "message": log_data["message"],
}
```

**Fix required:**
```python
broadcast_data = {
    "timestamp": log_data["timestamp"],
    "level": log_data["level"],
    "event_type": log_data["event_type"],
    "logger": log_data["event_type"],  # Add this line for LogViewer compatibility
    "message": log_data["message"],
}
```

### Issue: LogViewer Level Filter End-to-End
**Problem**: Need to verify the level filter dropdown in LogViewer works correctly end-to-end from UI → API → Database.

**Verification needed**:
- Selecting "ERROR" in dropdown sends `?level=ERROR` to API
- API filters by EventLog.level == "ERROR"
- Only ERROR logs returned and displayed

## 11. Time Range Filter

### Problem: Need to Filter Logs by Time

Users need a way to quickly filter logs to a specific time window to:
- Focus on recent issues (e.g., "what happened in the last hour")
- Reduce the number of logs returned for better performance
- Investigate incidents within a specific time range

### Solution: Predefined Time Range Options

**Option**: Add a time range dropdown in LogViewer with predefined options:

| Value | Label | Hours |
|-------|-------|-------|
| `all` | All Time | No filter |
| `1h` | Last 1 Hour | 1 |
| `4h` | Last 4 Hours | 4 |
| `8h` | Last 8 Hours | 8 |
| `1d` | Last 1 Day | 24 |
| `7d` | Last 7 Days | 168 |

### Backend (Already Implemented)

The backend `/api/logs` endpoint already supports time filtering:

**Query Parameters:**
- `since`: ISO 8601 timestamp to filter logs from this time
- `until`: ISO 8601 timestamp to filter logs until this time

**Location**: `bot/web/routes/logs.py` lines 76-77, 125-133

**Current implementation:**
```python
@router.get("/logs")
async def get_logs(
    since: Optional[str] = Query(None, description="ISO timestamp to filter logs since"),
    until: Optional[str] = Query(None, description="ISO timestamp to filter logs until"),
    ...
):
    # SQL filtering already exists:
    if since:
        since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
        filters.append(EventLog.timestamp >= since_dt)
    if until:
        until_dt = datetime.fromisoformat(until.replace('Z', '+00:00'))
        filters.append(EventLog.timestamp <= until_dt)
```

### Frontend Implementation Required

**Component**: `web/src/components/LogViewer.tsx`

**Changes needed:**
1. Add `timeRange` state (default: `'all'`)
2. Add `timeRange` to `filtersRef` to avoid stale closure
3. Update `fetchLogs()` to calculate `since` parameter from timeRange
4. Add time range dropdown in the filter row (use existing Select component)
5. Add `timeRange` to useEffect dependencies

**Time mapping:**
```typescript
const timeRangeHours: Record<string, number> = {
  'all': 0,
  '1h': 1,
  '4h': 4,
  '8h': 8,
  '1d': 24,
  '7d': 168,
}

// In fetchLogs():
if (timeRange !== 'all' && timeRangeHours[timeRange]) {
  const since = new Date(
    Date.now() - timeRangeHours[timeRange] * 60 * 60 * 1000
  ).toISOString()
  params.since = since
}
```

### Design Decision: Backend Already Ready

The backend was designed with time filtering from the start (even if not exposed via UI yet). This makes the frontend implementation straightforward - just need to:
1. Add the UI control
2. Pass the `since` parameter to the existing API

### Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Time zone handling | Medium | Use ISO 8601 with Z suffix (UTC) |
| Large datasets still slow | Low | Time filter reduces data, but consider pagination if needed |
| Stale filter values | Medium | Already solved with filtersRef pattern |

## Migration Plan

1. Add `StructuredFormatter` class to `llmcord.py`
2. Update `logging.basicConfig` to use new formatter
3. Update `log_handler.py` to add structured fields
4. Add environment variables to deployment config
5. Test in development environment first
6. Deploy to production with `ENVIRONMENT=production`

---

## 10. DEBUG Log Suppression (Performance Optimization)

### Problem: DEBUG Log Flood

During production testing, we discovered that DEBUG-level logging generates an excessive volume of logs, causing:

1. **Disk I/O Bottleneck**: ~200,000+ DEBUG log entries per hour being written to the database
2. **Database Bloat**: The `event_logs` table grows rapidly, impacting query performance
3. **Slow Portal Loading**: LogViewer becomes unresponsive when fetching from a large log table
4. **Resource Exhaustion**: High CPU usage from JSON serialization of debug messages

### Root Cause Analysis

#### 10.1.1 SQLAlchemy Internal Logs

The most significant source of DEBUG noise is SQLAlchemy's internal logging. These logs appear with our logger name ("discord-bot") but contain SQLAlchemy keywords:

```
2026-03-22T14:08:18.885Z [DEBUG] discord-bot | executing <function connect.<locals>.connector at 0x76e7d8682f20>
2026-03-22T14:08:18.887Z [DEBUG] discord-bot | executing <function connect.<locals>.connector at 0x76e7d8683ec0>
2026-03-22T14:08:18.889Z [DEBUG] discord-bot | operation <function connect.<locals>.connector> completed
```

**Keywords to filter:**
- `executing`
- `operation `
- `connect.<locals>`
- `create_function`
- `regexp`
- `sqlite3`

**Why it's proxied through our logger:** Python's logging hierarchy means third-party library logs bubble up to the root handler unless explicitly suppressed.

#### 10.1.2 Discord.py Library Logs

Discord.py (the Discord library) generates high-volume DEBUG logs for:
- Gateway events (connection, reconnection, heartbeat)
- API rate limiting
- Message processing
- Voice state changes

These are not currently suppressed and can generate thousands of entries per hour during active use.

#### 10.1.3 Application-Level DEBUG Logs

The following DEBUG logs exist in application code:

| File | Line | Message | Volume |
|------|------|---------|--------|
| `llmcord.py` | 729 | `MSG_SKIP: msg_id=...` | High - every ignored message |
| `llmcord.py` | 800 | `STT not configured, voice messages will be ignored` | Low |
| `llmcord.py` | 920 | `REPLY_SENT: ...` (now WARNING level - already improved) | Medium |
| `bot/llm/tools/registry.py` | 113 | Tool registry info | Low |
| `bot/llm/tools/yahoo_finance.py` | 54,80,108,119 | Stock price data | Medium |

### Solution Design

#### 10.2.1 Configuration-Level Suppression

**Remove DEBUG from portal.logs.levels:**

```yaml
# config.yaml
portal:
  logs:
    retention_days: 7
    levels:
    - INFO      # DEBUG removed to prevent DB flood
    - WARNING
    - ERROR
```

**Rationale:** The LogViewer primarily serves operational purposes (ERRORs, warnings). DEBUG logs were causing more harm than value.

#### 10.2.2 Log Handler Filter (First Line of Defense)

Add a `filter()` method to `DatabaseLogHandler` class in `bot/web/log_handler.py` to block noise before it reaches `emit()`:

```python
def filter(self, record: logging.LogRecord) -> logging.LogRecord | None:
    """
    First-line filter to block SQLAlchemy/asyncio DEBUG noise BEFORE emit is called.
    This prevents catastrophic I/O from database connection debug messages.
    """
    # Block ALL DEBUG logs that contain SQLAlchemy keywords
    # These logs have our logger name "discord-bot" but contain SQLAlchemy messages
    if record.levelno == logging.DEBUG:
        try:
            msg = record.getMessage()
            # These keywords indicate SQLAlchemy internal operations
            if any(kw in msg for kw in (
                "executing", "operation ", "connect.<locals>",
                "create_function", "regexp", "sqlite3"
            )):
                return None  # Drop the record
        except Exception:
            pass
    
    return record  # Allow all other logs
```

**Rationale:** The `filter()` method is called before `emit()` - dropping logs here avoids any I/O overhead.

#### 10.2.3 Third-Party Library Suppression

Add suppression for discord.py and other chatty libraries in `llmcord.py`:

```python
# Suppress noisy third-party DEBUG logs
# This prevents SQLAlchemy/asyncio from flooding logs with connection debug messages
for noisy_logger in [
    "sqlalchemy.engine", 
    "sqlalchemy.pool", 
    "asyncio",
    "discord",           # Discord.py library
    "discord.gateway",   # Discord gateway events
    "discord.http",      # Discord HTTP requests
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
```

**Locations in llmcord.py:**
- Line ~40: Early suppression (before structured logging setup)
- Line ~203: Suppression after structured logging is configured

#### 10.2.4 Application DEBUG Downgrades (Optional)

Consider downgrading these application DEBUG logs to INFO or removing them:

| Current | Proposed | Reason |
|---------|----------|--------|
| `logging.debug("MSG_SKIP: ...")` | Remove or use TRACE | Too verbose, fires on every ignored message |
| `logging.debug("STT not configured")` | `logging.info()` | Low volume, informative |
| `logging.debug("REPLY_SENT: ...")` | Remove or use TRACE | Too verbose, fires on every reply |
| `logging.debug("yahoo_finance: ...")` | `logging.info()` | Useful for debugging stock prices |

### Implementation Priority

| Priority | Task | Impact |
|----------|------|--------|
| P0 | Remove DEBUG from portal.logs.levels | Immediately stops DB flood |
| P0 | Add filter() method to DatabaseLogHandler | Blocks SQLAlchemy noise |
| P1 | Add discord.py suppression | Major reduction in volume |
| P2 | Downgrade application DEBUGs to INFO | Cleaner, more useful logs |
| P3 | Remove verbose MSG_SKIP/REPLY_SENT | Further volume reduction |

### Configuration Summary

After implementation:

```yaml
# config.yaml - Portal logs now only capture operational levels
portal:
  logs:
    levels:
    - INFO
    - WARNING
    - ERROR

# Environment variables for additional control
LOG_LEVEL=INFO           # Root log level (default: INFO in production)
LOG_SAMPLING_RATE=1.0    # Sampling rate (0.0-1.0)
ENVIRONMENT=production   # Set to "development" for DEBUG
DEBUG=true               # ONLY when actively debugging (enables httpx DEBUG)
```

### Trade-offs

| Benefit | Cost |
|---------|------|
| Massive I/O reduction | Cannot see DEBUG logs in LogViewer |
| Faster portal loading | Need to check console for DEBUG if needed |
| Smaller database | - |
| Cleaner operational logs | Some debugging context lost |

### Open Questions

1. Should DEBUG logs be available via console but not stored in DB?
2. Add a "DEBUG mode" toggle in the portal that temporarily enables DEBUG logging?
3. Implement log rotation for the database to auto-purge old logs?
4. Add a "Download DEBUG logs" feature that exports to a separate file?

---

## Migration (for existing deployments)

After deploying this change:

1. **Database**: Existing DEBUG logs remain but won't grow
2. **Config**: Update config.yaml to remove DEBUG from portal.logs.levels
3. **Console**: DEBUG logs still appear in console (unless DEBUG env var is removed)
4. **Portal**: LogViewer shows INFO, WARNING, ERROR only

To temporarily enable DEBUG in portal (for debugging):
```yaml
portal:
  logs:
    levels:
    - DEBUG    # Only enable temporarily!
    - INFO
    - WARNING
    - ERROR
```

---

## Open Questions

1. Should we add a migration guide for existing log queries in the portal?
2. Do we want to expose log sampling configuration via the web portal?
3. Should we add log level configuration per module (more granular than current)?
4. Add a "Download all logs" feature for DEBUG-level analysis?

---

## 12. Task Split-View Interface

### Problem: Inefficient Task Management Workflow

Current task management requires full-page navigation:
1. TaskList displays all tasks on full page
2. Click "Edit" → navigates to separate TaskEditor page
3. Edit YAML in raw text mode
4. Save → returns to list

This is inefficient and differs from the ConfigEditor experience which uses split-view.

### Solution: Split-View with Structured Forms

Convert to a two-column split view:
- **Left panel**: TaskList (always visible, scrollable)
- **Right panel**: TaskEditor (loads selected task dynamically)

Replace raw YAML editing with structured form fields.

### Design Goals

| Goal | Implementation |
|------|----------------|
| No page navigation | Always render both panels, use state for selection |
| Consistent with ConfigEditor | Use same split-view pattern |
| Better UX | Structured fields instead of raw YAML |
| Mobile support | Responsive: stack on mobile, split on desktop |

### Component Changes

#### 12.1 Dashboard.tsx

Replace conditional rendering with split-view:

```tsx
// Current (full-page toggle):
{editingTask !== undefined && editingTask !== null ? (
  <TaskEditor ... />
) : (
  <TaskList ... />
)}

// New (split-view):
<div className="flex flex-1 min-h-0 overflow-hidden">
  <div className="w-1/2 overflow-auto">
    <TaskList 
      selectedTask={selectedTask} 
      onSelectTask={setSelectedTask} 
    />
  </div>
  <div className="w-1/2 border-l overflow-auto">
    <TaskEditor 
      taskName={selectedTask} 
      onClose={() => setSelectedTask(null)} 
    />
  </div>
</div>
```

#### 12.2 TaskList.tsx

Add selection highlighting:
- New prop: `selectedTask?: string | null`
- Visual indicator: highlighted border/color for selected card
- "Add New" button sets `selectedTask` to empty string (new task mode)

#### 12.3 TaskEditor.tsx

Convert from Modal to side panel:
- New prop: `isPanel?: boolean` (default: true for split-view)
- Remove Modal wrapper in panel mode
- Add structured form fields
- Add Save/Cancel buttons

### Structured Form Fields

| Field | Type | Validation | Required |
|-------|------|------------|----------|
| name | Input | Unique, alphanumeric + hyphen | Yes |
| cron | Input | 5-field cron expression | Yes |
| enabled | Checkbox | Boolean | Yes |
| model | Input/Select | From config models | Yes |
| persona | Input | From config personas | No |
| prompt | Textarea | Non-empty string | Yes |
| tools | Multi-select | From tools.yaml | No |
| user_id | Input | Discord ID (string) | One of |
| channel_id | Input | Discord ID (string) | user_id/channel_id |

### Cron Validation

Client-side validation pattern:
```typescript
const CRON_REGEX = /^(\S+\s+){4}\S+$/

const validateCron = (cron: string): boolean => {
  return CRON_REGEX.test(cron.trim())
}
```

Common formats with descriptions:
| Cron | Description |
|------|-------------|
| `0 * * * *` | Every hour at minute 0 |
| `30 0 * * *` | Daily at 00:30 |
| `0 0 * * 0` | Weekly on Sunday at midnight |
| `*/15 * * * *` | Every 15 minutes |
| `0 9 * * 1-5` | Weekdays at 9 AM |

### Responsive Behavior

Using Tailwind breakpoints:
```tsx
// Mobile (<768px): stacked, editor as overlay
<div className="flex flex-col md:flex-row">
  {/* List: full width on mobile */}
  <div className="w-full md:w-1/2">TaskList</div>
  
  {/* Editor: full width on mobile, side-by-side on desktop */}
  <div className="w-full md:w-1/2 md:border-l">
    {selectedTask ? <TaskEditor /> : <EmptyState />}
  </div>
</div>
```

| Breakpoint | Layout | Behavior |
|------------|--------|----------|
| Mobile (<768px) | Stacked | List default, editor as full-screen overlay |
| Tablet (768-1023px) | Split 40%/60% | Both visible, narrower list |
| Desktop (≥1024px) | Split 50%/50% | Both visible equally |

### Backend Integration

Reuse existing API endpoints:
- **GET /api/tasks** - List all tasks
- **GET /api/tasks/{name}** - Get task config
- **PUT /api/tasks/{name}** - Update existing task
- **POST /api/tasks** - Create new task

Frontend converts form data to YAML before API call:
```typescript
const saveTask = async (formData: TaskFormData) => {
  const yaml = jsYaml.dump(formData)
  await axios.put(`/api/tasks/${formData.name}`, { config: yaml })
}
```

### Visual Mockup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tasks                                                                     │
├─────────────────────────────────┬───────────────────────────────────────────┤
│  ┌───────────────────────────┐  │  ┌─────────────────────────────────────┐│
│  │ [+ Add New] [⟳ Refresh]  │  │  │ Task Editor                    [X]  ││
│  ├───────────────────────────┤  │  ├─────────────────────────────────────┤│
│  │ ▼ sample-task    [ON]  🖊️│  │  │ Name:    [sample-task           ]    ││
│  │   [▶ Run] [🖊️ Edit]      │  │  │ Cron:    [30 0 * * 1-6          ]    ││
│  ├───────────────────────────┤  │  │ Enabled: [x]                        ││
│  │ ▶ another-task  [OFF] 🖊️ │  │  │ Model:   [google/gemini-2.5-flash]   ││
│  │ ▶ third-task    [ON]  🖊️ │  │  │ Persona: [stock_market_analyst ]     ││
│  └───────────────────────────┘  │  │ Prompt:  [                      ]    ││
│                                 │  │           [                      ]    ││
│  [Selected: border-primary]     │  │ Tools:   [✓ web_search     ]         ││
│                                 │  │           [✓ get_market_prices]      ││
│                                 │  │ User ID: [1234567890            ]    ││
│                                 │  │ Channel: [0987654321            ]    ││
│                                 │  ├─────────────────────────────────────┤│
│                                 │  │ [Cancel]           [Save Task]      ││
│                                 │  └─────────────────────────────────────┘│
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### Trade-offs

| Benefit | Cost |
|---------|------|
| Faster workflow (no page navigation) | More complex component state |
| Structured input reduces errors | More form validation code |
| Consistent with ConfigEditor | Responsive design complexity |
| Better mobile experience | |

### Open Questions

1. Should empty state show a "Select a task or create new" message?
2. Confirm tool list should come from `/api/config` or `/api/tools`?
3. Add auto-save draft before closing editor?
4. Support keyboard shortcuts (Ctrl+S to save, Esc to close)?