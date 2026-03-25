# Logging Configuration

This guide covers the structured logging system, log levels, and configuration options.

## Log Levels

| Level | Usage | Stored in DB |
|-------|-------|--------------|
| DEBUG | Development only, detailed diagnostics | ❌ No |
| INFO | Normal operations, startup, user actions | ✅ Yes |
| WARNING | Unexpected but recoverable | ✅ Yes |
| ERROR | Failures requiring investigation | ✅ Yes |
| CRITICAL | Severe issues, system-wide problems | ✅ Yes |

> **Note**: DEBUG logs are not stored in the database to prevent I/O flooding from third-party libraries (SQLAlchemy, discord.py).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Deployment environment (`development` or `production`) |
| `LOG_LEVEL` | `INFO` (prod) / `DEBUG` (dev) | Override log level |
| `LOG_SERVICE` | `discord-bot` | Service identifier in logs |
| `LOG_SAMPLING_RATE` | `1.0` | Log sampling rate (0.0-1.0) for production |

### ENVIRONMENT

- **development**: Human-readable console output
- **production**: JSON structured logs

```bash
# Set environment
export ENVIRONMENT=production
```

### LOG_LEVEL

Overrides both config and default levels. Highest priority.

```bash
# Only show WARNING and above
export LOG_LEVEL=WARNING
```

### LOG_SERVICE

Sets the service name in log output (useful for microservices).

```bash
export LOG_SERVICE=my-bot
```

### LOG_SAMPLING_RATE

Reduces log volume in high-traffic production environments.

```bash
# Log only 10% of INFO-level messages
export LOG_SAMPLING_RATE=0.1
```

> Errors (WARNING, ERROR, CRITICAL) are always logged regardless of sampling rate.

---

## Configuration (config.yaml)

### Log Level

```yaml
behavior:
  # Can be overridden by LOG_LEVEL environment variable
  log_level: INFO
```

### Portal Log Retention

```yaml
portal:
  logs:
    retention_days: 7
    # DEBUG logs cause massive I/O - only enable temporarily for debugging
    levels:
    - INFO
    - WARNING
    - ERROR
```

> Adding DEBUG to `portal.logs.levels` is not recommended in production as it generates ~200k+ entries/hour from third-party libraries.

---

## Domain Loggers

The system uses domain-specific loggers for better filtering:

| Logger | Usage |
|--------|-------|
| `discord-bot.discord` | Discord bot events, commands, message handling |
| `discord-bot.api` | HTTP API requests and responses |
| `discord-bot.auth` | Authentication events |
| `discord-bot.llm` | LLM interactions (prompts, responses, tool calls) |
| `discord-bot.voice` | Voice operations (STT, TTS) |
| `discord-bot.process` | Main process, scheduler, startup/shutdown |
| `discord-bot.db` | Database operations |

### Using Domain Loggers

```python
from llmcord import get_logger

# Get a domain-specific logger
api_logger = get_logger("api")
llm_logger = get_logger("llm")

# Use them
api_logger.info("Request received", extra={"request_id": "abc123"})
llm_logger.info("Generating response", extra={"user_id": "user456"})
```

---

## Structured Log Format

### Production (JSON)

```json
{
  "timestamp": "2026-03-23T10:30:00.123Z",
  "level": "INFO",
  "message": "User message received",
  "service": "discord-bot",
  "environment": "production",
  "trace_id": "abc123",
  "user_id": "user456"
}
```

### Development (Human-Readable)

```
2026-03-23T10:30:00.123Z [INFO] discord-bot | User message received
```

---

## LogViewer (Web Portal)

The web portal's LogViewer displays structured logs:

- **Filter by level**: Select INFO/WARNING/ERROR from dropdown
- **Filter by service**: Filter by service name
- **Filter by environment**: Filter by development/production
- **Expandable rows**: View trace_id, request_id, user_id

---

## Sensitive Data Handling

Never log:
- Passwords and secrets
- API keys and tokens
- Credit card numbers
- National IDs
- Personal identifiable information (PII)

---

## Related

- [Getting Started](getting-started.md)
- [Configure Providers](configure-provider.md)