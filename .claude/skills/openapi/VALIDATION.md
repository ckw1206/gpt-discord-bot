# OpenAPI Validation

## Specification Completeness

- All endpoints have descriptions
- Request/response schemas defined
- Examples provided for clarity
- Security requirements specified
- Error responses documented
- Parameters have descriptions and types
- Tags used for organization

## Schema Design

- Use appropriate JSON Schema types
- Define required vs optional fields
- Add format constraints (email, date, uuid)
- Set min/max length and value constraints
- Use enums for fixed value sets
- Avoid overly permissive schemas (e.g., additionalProperties: true)
- Reference shared schemas with $ref

## REST Compliance

- Resource-oriented URLs (nouns not verbs)
- Correct HTTP method usage
  - GET for retrieval (no body)
  - POST for creation
  - PUT for full replacement
  - PATCH for partial updates
  - DELETE for removal
- Proper status codes
  - 200 OK for successful GET
  - 201 Created for POST
  - 204 No Content for DELETE
  - 400 Bad Request for validation errors
  - 401 Unauthorized for auth failures
  - 404 Not Found for missing resources

## Security

- Security schemes defined (OAuth2, API Key, JWT)
- Security requirements applied to endpoints
- Scopes documented for OAuth2
- HTTPS enforced in servers section
