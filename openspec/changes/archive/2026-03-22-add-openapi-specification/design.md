## Context

The GPT Discord Bot has ~30 API endpoints across 8 route files + server.py, but lacks proper OpenAPI documentation. FastAPI auto-generates a basic schema, but it's incomplete:
- No custom tags for API grouping
- No security scheme defined
- No error response documentation
- No request/response examples
- Minimal endpoint descriptions

The API is currently internal-use only (web portal), but proper documentation would enable future third-party integrations.

## Goals / Non-Goals

**Goals:**
- Add comprehensive OpenAPI specification accessible at `/openapi.json`
- Enable Swagger UI at `/docs` for API exploration (internal)
- Document all 30+ endpoints with summaries and descriptions
- Define error response schemas for consistency
- Add OAuth2 bearer token security scheme
- Group endpoints by tags (auth, config, tasks, status, skills, servers, personas, logs)

**Non-Goals:**
- Do NOT expose API to external public users (remains internal)
- Do NOT add rate limiting or API keys (different concern)
- Do NOT generate client code (future enhancement)
- Do NOT add new API endpoints (documentation only)

## Decisions

### 1. Where to add OpenAPI configuration?

**Decision**: Add to `bot/web/server.py` where the FastAPI app is defined.

**Rationale**: The app is already created in `server.py`, and that's where the title, description, and version are set. Adding tags and security schemes there keeps configuration centralized.

### 2. Error schema approach?

**Decision**: Create shared error schemas in a dedicated module.

**Rationale**: Instead of repeating error definitions across endpoints, create centralized error response models that can be reused. This follows DRY principles.

**Alternative considered**: Inline error documentation per endpoint. Rejected because it's repetitive and harder to maintain.

### 3. Security scheme type?

**Decision**: Use OAuth2 password bearer flow.

**Rationale**: The API already uses token-based authentication. OAuth2 bearer token is the standard way to document this in OpenAPI.

### 4. Endpoint documentation strategy?

**Decision**: Add docstrings to endpoints, leverage existing Pydantic models.

**Rationale**: FastAPI already uses Pydantic for request/response models. The schemas are already generated from these. The main work is adding descriptions and examples.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Documentation drift | API changes but spec not updated | Add OpenAPI validation to CI/tests |
| Large spec file | Hard to navigate | Use tags to group, keep descriptions concise |
| Sensitive data exposure | API docs might reveal internals | Keep Swagger UI internal-only (auth required) |

## Migration Plan

1. **Phase 1 - Foundation** (Tasks 1-2):
   - Add openapi_tags to server.py
   - Define security scheme

2. **Phase 2 - Core Endpoints** (Tasks 3-4):
   - Add summaries/descriptions to high-use endpoints (config, status, auth)
   - Document common error responses

3. **Phase 3 - Remaining Endpoints** (Task 5):
   - Document tasks, skills, servers, personas, logs endpoints

4. **Phase 4 - Verification** (Task 6):
   - Verify /openapi.json generates correctly
   - Verify /docs works with authentication

## Open Questions

1. Should the OpenAPI spec be published separately (e.g., as a static file)?
2. Should we add a version field to the spec for API versioning?
3. Do we want to generate TypeScript types from the spec (future)?