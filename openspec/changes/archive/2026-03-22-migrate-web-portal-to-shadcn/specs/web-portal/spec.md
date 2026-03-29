## ADDED Requirements

### Requirement: Web portal uses shadcn/ui component library
The web portal SHALL use shadcn/ui components for all UI elements, providing consistent, accessible, and maintainable components.

#### Scenario: Components migrated to shadcn
- **WHEN** the web portal renders any UI element
- **THEN** it uses shadcn/ui components (Button, Input, Dialog, Table, Tabs, Select, Card, etc.)
- **AND** follows shadcn composition patterns (FieldGroup, Field, etc.)
- **AND** uses lucide-react for icons

## MODIFIED Requirements

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