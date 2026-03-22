---
name: logging-guide
description: |
  Implement structured logging with proper log levels and sensitive data handling.
  Use when: adding logging, debugging, setting up observability.
  Keywords: logging, log level, structured logging, observability, logger.
source: ../../../../skills/logging-guide/SKILL.md
source_version: 1.0.0
translation_version: 1.0.1
last_synced: 2026-01-08
status: current
scope: universal
---

# Logging Guide

> **Language**: English | [繁體中文](./SKILL.zh-tw.md)

**Version**: 1.0.0
**Last Updated**: 2025-12-30
**Scope**: Claude Code Skills

---

## Purpose

This skill helps implement consistent, structured, and actionable application logging across all environments.

## Quick Reference

### Log Levels

| Level | Code | When to Use | Production |
|-------|------|-------------|------------|
| **TRACE** | 10 | Very detailed debug info | Off |
| **DEBUG** | 20 | Detailed debug info | Off |
| **INFO** | 30 | Normal operation events | On |
| **WARN** | 40 | Recoverable issues | On |
| **ERROR** | 50 | Errors needing attention | On |
| **FATAL** | 60 | Critical failures | On |

### Level Selection Decision Tree

```
Only for debugging?         → DEBUG (off in production)
Normal operation complete?  → INFO
Unexpected but okay?        → WARN
Operation failed?           → ERROR
App can't continue?         → FATAL
```

### When to Use Each Level

| Level | Examples |
|-------|----------|
| **TRACE** | Function entry/exit, loop iterations, variable values |
| **DEBUG** | State changes, config values, query parameters |
| **INFO** | App startup/shutdown, user actions, scheduled tasks |
| **WARN** | Deprecated API, retry attempts, resources near limit |
| **ERROR** | Failed operations, caught exceptions, integration failures |
| **FATAL** | Unrecoverable errors, startup failures, lost critical resources |

## Structured Logging

### Required Fields

```json
{
  "timestamp": "2025-01-15T10:30:00.123Z",
  "level": "INFO",
  "message": "User login successful",
  "service": "auth-service",
  "environment": "production"
}
```

### Recommended Fields

```json
{
  "timestamp": "2025-01-15T10:30:00.123Z",
  "level": "INFO",
  "message": "User login successful",
  "service": "auth-service",
  "environment": "production",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "usr_12345",
  "request_id": "req_67890",
  "duration_ms": 150,
  "http_method": "POST",
  "http_path": "/api/v1/login",
  "http_status": 200
}
```

### Field Naming Conventions

Use `snake_case` with domain prefixes:

| Domain | Common Fields |
|--------|---------------|
| HTTP | http_method, http_path, http_status, http_duration_ms |
| Database | db_query_type, db_table, db_duration_ms, db_rows_affected |
| Queue | queue_name, queue_message_id, queue_delay_ms |
| User | user_id, user_role, user_action |
| Request | request_id, trace_id, span_id |

## Detailed Guide

For complete standards, see:
- [Logging Standards](../../core/logging-standards.md)

### AI-Optimized Format (Token Savings)

AI assistants can use YAML format files to reduce token usage:
- Base standard: `ai/standards/logging.ai.yaml`

## Sensitive Data Handling

### Never Log

- Passwords or secrets
- API keys or tokens
- Credit card numbers
- National ID numbers
- Full authentication tokens

### Mask or Redact

```javascript
// Bad
logger.info('Login attempt', { password: userPassword });

// Good
logger.info('Login attempt', { password: '***REDACTED***' });

// Good - partial mask
logger.info('Card processed', { last_four: '4242' });
```

### PII Handling

- Log user IDs instead of email when possible
- Use hashed identifiers for sensitive queries
- Set data retention policies

## Error Logging

### Required Error Fields

```json
{
  "level": "ERROR",
  "message": "Database connection failed",
  "error_type": "ConnectionError",
  "error_message": "Connection refused",
  "error_code": "ECONNREFUSED",
  "stack": "Error: Connection refused\n    at connect (/app/db.js:45:11)..."
}
```

### Error Context

Always include:
- What operation was attempted
- Relevant IDs (user_id, request_id)
- Input parameters (sanitized)
- Retry count (if applicable)

```javascript
logger.error('Order processing failed', {
  error_type: err.name,
  error_message: err.message,
  order_id: orderId,
  user_id: userId,
  retry_count: 2,
  stack: err.stack
});
```

## Log Formats

### JSON Format (Production)

```json
{"timestamp":"2025-01-15T10:30:00.123Z","level":"INFO","message":"Request completed","request_id":"req_123","duration_ms":45}
```

### Human-Readable Format (Development)

```
2025-01-15T10:30:00.123Z [INFO] Request completed request_id=req_123 duration_ms=45
```

## Performance Considerations

### Log Volume by Environment

| Environment | Level | Strategy |
|-------------|-------|----------|
| Development | DEBUG | All logs |
| Testing | INFO | Most logs |
| Production | INFO | Sample high-traffic endpoints |

### High-Traffic Endpoints

- Use sampling (1 in 100)
- Aggregate metrics instead of individual logs
- Use separate log streams

## Checklist

### Required Fields

- [ ] timestamp (ISO 8601)
- [ ] level
- [ ] message
- [ ] service name
- [ ] request_id or trace_id

### Security

- [ ] No passwords or secrets
- [ ] No full tokens
- [ ] PII masked or hashed
- [ ] Credit cards never logged
- [ ] Retention policy set

---

## Configuration Detection

This skill supports project-specific settings.

### Detection Order

1. Check existing logging library configuration
2. Check logging guide in `CONTRIBUTING.md`
3. If not found, **default to structured JSON logging**

### First-Time Setup

If no logging standards found:

1. Suggest: "This project doesn't have logging standards set up. Would you like to set up structured logging?"
2. Suggest documenting in `CONTRIBUTING.md`:

```markdown
## Logging Standards

### Log Levels
- DEBUG: Development only, detailed diagnostic info
- INFO: Normal operations (startup, user actions, tasks)
- WARN: Unexpected but recoverable situations
- ERROR: Failures that need investigation

### Required Fields
All logs must include: timestamp, level, message, service, request_id

### Sensitive Data
Never log: passwords, tokens, credit cards, national IDs
```

---

## Related Standards

- [Logging Standards](../../core/logging-standards.md)
- [Error Code Standards](../../core/error-code-standards.md)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-30 | Initial release |

---

## License

This skill is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Source**: [universal-dev-standards](https://github.com/AsiaOstrich/universal-dev-standards)
