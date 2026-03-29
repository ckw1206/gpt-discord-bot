## Why

The GPT Discord Bot has ~30 API endpoints across 8 route files + server.py (auth, config, tasks, status, skills, servers, personas, logs), but lacks proper OpenAPI documentation. While FastAPI auto-generates a basic OpenAPI schema at `/openapi.json`, it lacks proper tags, security schemes, error response documentation, and request/response examples. This limits the API's usefulness for:
- Third-party integrations
- Frontend development (API contract)
- Developer on-boarding

## What Changes

- Add OpenAPI tags to the main FastAPI app for all route groups
- Define OAuth2 password bearer security scheme
- Add summary and description to all major endpoints
- Document error responses for endpoints
- Add request/response examples where helpful
- Configure Swagger UI access (internal use)
- Add centralized error schemas

## Capabilities

### New Capabilities
- `api-documentation`: Comprehensive OpenAPI specification for all web portal APIs, including security, error schemas, and examples

### Modified Capabilities
- `web-portal`: The existing web-portal capability will have its API formally documented via OpenAPI spec (requirements unchanged, just better developer experience)

## Impact

- **Files modified**: `bot/web/server.py` (main app configuration + auth endpoints)
- **Files enhanced**: All route files in `bot/web/routes/` (add endpoint docs)
- **New files**: `bot/web/schemas.py` (shared error response models)
- **Dependencies**: No new external dependencies (FastAPI already includes OpenAPI support)