## Purpose
Provide a web-based admin portal for managing the Discord bot.
## Requirements
### Requirement: Web portal serves on configurable port
The system SHALL provide a web portal accessible at a configurable port (default 8080) that allows users to monitor and configure the Discord bot.

#### Scenario: Portal accessible on default port
- **WHEN** portal is enabled and PORT env var is not set
- **THEN** portal is accessible at http://localhost:8080

#### Scenario: Portal accessible on custom port
- **WHEN** PORT environment variable is set to 9000
- **THEN** portal is accessible at http://localhost:9000

### Requirement: First-time setup wizard
The system SHALL provide a setup wizard on first login to create an admin password when no users exist in the database.

#### Scenario: First login shows setup wizard
- **WHEN** no users exist in DB and user accesses /login
- **THEN** setup wizard UI is shown to create initial admin user

#### Scenario: Setup wizard creates user
- **WHEN** POST /api/auth/setup is called with username and password
- **THEN** user is created with bcrypt-hashed password, JWT token returned

#### Scenario: Setup wizard blocked after users exist
- **WHEN** POST /api/auth/setup is called when users already exist
- **THEN** HTTP 403 is returned

### Requirement: Authentication required for portal access
The system SHALL require authentication before allowing access to any portal API endpoint.

#### Scenario: Unauthenticated request rejected
- **WHEN** a request is made to /api/* without valid JWT
- **THEN** HTTP 401 is returned

#### Scenario: Authenticated request allowed
- **WHEN** a request is made with valid JWT token
- **THEN** HTTP 200 is returned with requested data

### Requirement: Portal displays bot status
The system SHALL provide an endpoint that returns current bot status including online/offline state, server count, and uptime.

#### Scenario: Bot online returns status
- **WHEN** user requests /api/status while bot is connected
- **THEN** response includes: online: true, server_count: N, uptime_seconds: N

#### Scenario: Bot offline returns status
- **WHEN** user requests /api/status while bot is disconnected
- **THEN** response includes: online: false

### Requirement: Portal lists servers and channels
The system SHALL provide an endpoint that returns a list of connected Discord servers and their text channels.

#### Scenario: Returns server list
- **WHEN** user requests /api/servers
- **THEN** response includes array of servers with id, name, and text_channels

### Requirement: Simple config fields editable
The system SHALL allow editing of simple config fields: status_message, max_text, max_images, max_messages, allow_dms, use_plain_responses, show_embed_color.

**MODIFIED Description:** The system SHALL allow editing of simple config fields through a shadcn/ui-based form interface with proper validation and feedback.

#### Scenario: Update status_message
- **WHEN** user PUTs new status_message to /api/config
- **THEN** config.yaml is updated and bot reloads config

#### Scenario: Update max_text
- **WHEN** user PUTs new max_text value to /api/config
- **THEN** config.yaml is updated with new value

### Requirement: Persona list viewable
The system SHALL provide read-only access to list all available personas.

**MODIFIED Description:** The system SHALL provide access to list and manage personas through a shadcn/Table-based UI.

#### Scenario: List personas
- **WHEN** user requests GET /api/personas
- **THEN** response includes array of persona names

#### Scenario: Get persona content
- **WHEN** user requests GET /api/personas/{name}
- **THEN** response includes persona configuration

### Requirement: Task list viewable
The system SHALL provide read-only access to list all scheduled tasks.

#### Scenario: List tasks
- **WHEN** user requests GET /api/tasks
- **THEN** response includes array of task configurations

### Requirement: Refresh command available via API
The system SHALL provide an endpoint to trigger config reload equivalent to /refresh slash command.

#### Scenario: Trigger refresh via API
- **WHEN** user POSTs to /api/refresh
- **THEN** config is reloaded and response confirms success

### Requirement: Web portal uses shadcn/ui component library
The web portal SHALL use shadcn/ui components for all UI elements, providing consistent, accessible, and maintainable components.

#### Scenario: Components migrated to shadcn
- **WHEN** the web portal renders any UI element
- **THEN** it uses shadcn/ui components (Button, Input, Dialog, Table, Tabs, Select, Card, etc.)
- **AND** follows shadcn composition patterns (FieldGroup, Field, etc.)
- **AND** uses lucide-react for icons

### Requirement: Portal config section editable in ConfigEditor
The system SHALL allow editing of portal configuration fields (enabled, port, cors_origins, docs_enabled, logs) through the ConfigEditor UI.

#### Scenario: Portal section displays in ConfigEditor sidebar
- **WHEN** user navigates to Config tab
- **THEN** Portal section is visible in the sidebar navigation

#### Scenario: Portal enabled toggle works
- **WHEN** user toggles portal.enabled from true to false
- **AND** clicks Save
- **THEN** config.yaml is updated with portal.enabled: false

#### Scenario: Portal docs_enabled toggle works
- **WHEN** user toggles portal.docs_enabled from false to true
- **AND** clicks Save
- **THEN** config.yaml is updated with portal.docs_enabled: true

#### Scenario: Portal port field is editable
- **WHEN** user edits portal.port value to 9000
- **AND** clicks Save
- **THEN** config.yaml is updated with portal.port: 9000

#### Scenario: Portal logs retention_days is editable
- **WHEN** user edits portal.logs.retention_days value
- **AND** clicks Save
- **THEN** config.yaml is updated with new retention_days value

#### Scenario: Portal logs levels is editable
- **WHEN** user edits portal.logs.levels (comma-separated)
- **AND** clicks Save
- **THEN** config.yaml is updated with new levels array

### Requirement: Voice config section editable in ConfigEditor
The system SHALL allow editing of voice configuration fields (region, default_voice) through the ConfigEditor UI.

#### Scenario: Voice section displays in ConfigEditor sidebar
- **WHEN** user navigates to Config tab
- **THEN** Voice section is visible in the sidebar navigation

#### Scenario: Voice region field is editable
- **WHEN** user edits voice.region value
- **AND** clicks Save
- **THEN** config.yaml is updated with new region value

#### Scenario: Voice default_voice field is editable
- **WHEN** user edits voice.default_voice value
- **AND** clicks Save
- **THEN** config.yaml is updated with new default_voice value

#### Scenario: Voice key is not displayed (sensitive)
- **WHEN** user views Voice section
- **THEN** voice.key field is NOT shown to protect sensitive credentials

