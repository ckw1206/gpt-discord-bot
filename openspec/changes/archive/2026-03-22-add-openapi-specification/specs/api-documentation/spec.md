## ADDED Requirements

### Requirement: OpenAPI specification endpoint available
The system SHALL provide a valid OpenAPI 3.0 schema at `/openapi.json` that documents all REST API endpoints.

#### Scenario: OpenAPI endpoint accessible
- **WHEN** a client sends GET request to `/openapi.json`
- **THEN** the server returns a valid OpenAPI 3.0 JSON document with status 200

#### Scenario: OpenAPI contains all endpoints
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL include all documented endpoints from:
  - API routers in `bot/web/routes/` (config, tasks, status, skills, servers, personas, logs)
  - Direct endpoints in `bot/web/server.py` (auth: /api/auth/login, /api/auth/setup, /api/auth/has-users, /api/auth/users, health check, WebSocket)

#### Scenario: OpenAPI has valid info section
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL contain an `info` object with `title`, `version`, and `description` fields

### Requirement: Swagger UI available for API exploration
The system SHALL provide an interactive API documentation interface at `/docs` accessible to authenticated users.

#### Scenario: Swagger UI renders
- **WHEN** an authenticated user navigates to `/docs`
- **THEN** the Swagger UI interface loads with all documented endpoints

#### Scenario: Swagger UI uses bearer auth
- **WHEN** a user tries an endpoint requiring authentication in Swagger UI
- **THEN** they SHALL be able to input their bearer token to authenticate

### Requirement: Swagger UI can be disabled for security
The system SHALL provide a configuration option to enable/disable the Swagger UI documentation endpoints.

#### Scenario: Docs disabled in config
- **WHEN** `portal.docs_enabled` is set to `false` in config.yaml
- **THEN** GET `/docs` returns 404
- **THEN** GET `/redoc` returns 404
- **AND** GET `/openapi.json` returns 404 (to prevent schema leakage)

#### Scenario: Docs enabled in config (default)
- **WHEN** `portal.docs_enabled` is set to `true` or not specified in config.yaml
- **THEN** GET `/docs` returns the Swagger UI
- **THEN** GET `/redoc` returns the ReDoc interface

### Requirement: Endpoints grouped by tags
The OpenAPI specification SHALL group endpoints by functional area using tags.

#### Scenario: Tags defined for each route group
- **WHEN** the OpenAPI schema is generated
- **THEN** endpoints SHALL be tagged as: auth, config, tasks, status, skills, servers, personas, logs

### Requirement: Security scheme documented
The OpenAPI specification SHALL define the authentication mechanism used by the API.

#### Scenario: Bearer auth security scheme
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL include a security scheme of type `http` with scheme `bearer`

#### Scenario: Protected endpoints require auth
- **WHEN** the OpenAPI schema is generated
- **THEN** all endpoints except:
  - GET `/health` (health check)
  - POST `/api/auth/setup` (first-time setup only)
  - GET `/api/auth/has-users` (check if users exist)
  - GET `/openapi.json` (schema)
  - GET `/docs` and GET `/redoc` (Swagger UI)
  - SHALL require bearer authentication

### Requirement: Error responses documented
The OpenAPI specification SHALL document common error responses for endpoints.

#### Scenario: 401 Unauthorized documented
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL document 401 responses for protected endpoints

#### Scenario: 403 Forbidden documented
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL document 403 responses for endpoints with permission checks

#### Scenario: 404 Not Found documented
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL document 404 responses for endpoints that return single resources

#### Scenario: 500 Internal Server Error documented
- **WHEN** the OpenAPI schema is generated
- **THEN** it SHALL document 500 responses for all endpoints

### Requirement: Endpoint summaries and descriptions
All major endpoints SHALL have descriptive summaries in the OpenAPI spec.

#### Scenario: Config endpoints documented
- **WHEN** the OpenAPI schema is generated
- **THEN** GET `/api/config` SHALL have a summary "Get current configuration"
- **THEN** PUT `/api/config` SHALL have a summary "Update configuration values"

#### Scenario: Status endpoint documented
- **WHEN** the OpenAPI schema is generated
- **THEN** GET `/api/status` SHALL have a summary "Get bot status"

#### Scenario: Tasks endpoints documented
- **WHEN** the OpenAPI schema is generated
- **THEN** task-related endpoints SHALL have summaries describing their function

### Requirement: Response schemas use Pydantic models
The OpenAPI specification SHALL leverage existing Pydantic models for request/response validation.

#### Scenario: Response models reflected in schema
- **WHEN** an endpoint has a `response_model` defined
- **THEN** that model SHALL appear as a schema component in the OpenAPI spec

#### Scenario: Request models reflected in schema
- **WHEN** an endpoint accepts a request body with a Pydantic model
- **THEN** that model SHALL appear as a schema component in the OpenAPI spec

---

## Technical Implementation

This section provides technical implementation details for the OpenAPI documentation.

- **Title**: GPT Discord Bot Web Portal API
- **Version**: 1.0.0
- **Base URL**: `/api` (relative)
- **Schemes**: HTTP, HTTPS

## Authentication

OAuth2 password bearer token authentication is used for all protected endpoints.

```yaml
components:
  securitySchemes:
    bearerAuth:
      type: oauth2
      flows:
        password:
          tokenUrl: /api/auth/login
          scopes: {}
```

## Error Responses

All endpoints may return standard HTTP error responses:

| Status Code | Description | Schema |
|-------------|-------------|--------|
| 400 | Bad Request | HTTPError |
| 401 | Unauthorized | HTTPError |
| 403 | Forbidden | HTTPError |
| 404 | Not Found | NotFoundError |
| 500 | Internal Server Error | HTTPError |

## WebSocket Endpoints

WebSocket endpoints require an upgrade request with bearer token authentication.

### /ws/status

- **Purpose**: Real-time bot status updates
- **Protocol**: WebSocket (ws:// or wss://)
- **Authentication**: Bearer token in connection query param
- **Messages**: JSON status updates

### /ws/logs

- **Purpose**: Real-time log streaming
- **Protocol**: WebSocket (ws:// or wss://)
- **Authentication**: Bearer token in connection query param
- **Messages**: JSON log entries with level, timestamp, message

## OpenAPI Enhancements

### Operation IDs

Each endpoint should have a unique `operationId` for easier client code generation:

```yaml
paths:
  /api/status:
    get:
      operationId: getBotStatus
      summary: Get bot status
```

### Servers Field

Add servers configuration for different environments:

```yaml
servers:
  - url: http://localhost:8000
    description: Local development
  - url: https://api.example.com
    description: Production
```

## Endpoint Tags

Endpoints should be grouped by these tags:

| Tag | Endpoints |
|-----|-----------|
| Auth | login, setup, has-users, users |
| Config | config (GET, PUT) |
| Status | status, update-presence |
| Tasks | task operations |
| Skills | skill operations |
| Servers | server operations |
| Personas | persona operations |
| Logs | log operations |
| WebSocket | ws/status, ws/logs |

## Validation

The generated OpenAPI schema should pass validation using:

- Spectral (spectral lint)
- FastAPI's built-in validation
- Swagger Editor validation