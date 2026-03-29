## 1. Foundation - Server Configuration

- [x] 1.1 Add openapi_tags to FastAPI app in bot/web/server.py
- [x] 1.2 Define OAuth2 bearer security scheme in server.py

## 2. Error Schemas

- [x] 2.1 Create bot/web/schemas.py with shared error response models (HTTPError, NotFoundError, etc.)
- [x] 2.2 Add error responses to FastAPI app in server.py using add_api_route or decorator

## 3. Core Endpoints Documentation (routes/)

- [x] 3.1 Add summary/description to GET /api/config endpoint
- [x] 3.2 Add summary/description to PUT /api/config endpoint
- [x] 3.3 Add summary/description to GET /api/status endpoint

## 4. Auth Endpoints Documentation (server.py)

- [x] 4.1 Add summary/description to POST /api/auth/login endpoint (in server.py)
- [x] 4.2 Add summary/description to POST /api/auth/setup endpoint (in server.py)
- [x] 4.3 Add summary/description to GET /api/auth/has-users endpoint (in server.py)
- [x] 4.4 Add summary/description to GET /api/auth/users endpoint (in server.py)

## 5. Remaining Endpoints Documentation

- [x] 5.1 Add summaries to task endpoints (list, get, create, update, delete, run, reload, status)
- [x] 5.2 Add summaries to skill endpoints (list, detail)
- [x] 5.3 Add summaries to server endpoints (list, detail, permissions)
- [x] 5.4 Add summaries to persona endpoints (list, detail, create, update, delete, usage)
- [x] 5.5 Add summaries to log endpoints (list, levels, types)

## 5.6 Bot Status Endpoints Documentation

- [x] 5.6.1 Add summary/description to POST /api/bot/update-presence endpoint (status.py)
- [x] 5.6.2 Add summary/description to GET /api/personas/{name}/usage endpoint (personas.py)

## 5.7 WebSocket Endpoints Documentation

- [x] 5.7.1 Document WebSocket /ws/status endpoint with connection upgrade info
- [x] 5.7.2 Document WebSocket /ws/logs endpoint with connection upgrade info

## 6. Verification

- [x] 6.1 Start the web server and verify /openapi.json returns valid schema
- [x] 6.2 Verify /docs renders Swagger UI correctly
- [x] 6.3 Verify tags group endpoints properly in documentation
- [x] 6.4 Test authentication flow in Swagger UI

## 7. Docs Toggle Security Feature

- [x] 7.1 Add `docs_enabled` field to portal config schema in bot/web/config.py
- [x] 7.2 Add `docs_enabled` to config.yaml default (true for dev)
- [x] 7.3 Update server.py to conditionally enable /docs and /redoc based on config

## 8. Frontend Portal Config UI

- [x] 8.1 Add `portal.docs_enabled` to portal fields in web/src/components/ConfigEditor.tsx (CONFIG_SECTIONS)
- [x] 8.2 Add label mapping for 'Docs Enabled' in getFieldLabel function
- [x] 8.3 Verify toggle renders as checkbox in ConfigEditor UI
- [x] 8.4 Test saving docs_enabled value via API and config reload