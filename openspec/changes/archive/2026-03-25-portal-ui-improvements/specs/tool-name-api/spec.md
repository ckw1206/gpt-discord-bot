# Tool Name API

## ADDED Requirements

### Requirement: Tool names endpoint
The system SHALL provide an endpoint to return actual tool names from the registry.

#### Scenario: Get tool list
- **WHEN** client requests GET /api/tools
- **THEN** response includes array of {name, description, parameters} for each tool
- **AND** tool names match registry names (e.g., 'get_market_prices', not 'yahoo_finance')

#### Scenario: Get tool names only
- **WHEN** client requests GET /api/tools/names
- **THEN** response includes array of tool name strings only
- **AND** names match the actual callable tool names from registry

### Requirement: Tool schema in response
The /api/tools endpoint SHALL include schema information for each tool.

#### Scenario: Tool with parameters
- **WHEN** tool has parameters in its schema
- **THEN** response includes parameters array with {name, type, description}
- **AND** client can display parameter information in UI

### Requirement: Tool names used in TaskEditor
The TaskEditor SHALL use /api/tools endpoint for populating the tools dropdown.

#### Scenario: Load tools in TaskEditor
- **WHEN** TaskEditor component mounts
- **THEN** it fetches from /api/tools (not /api/skills)
- **AND** dropdown shows actual tool names like 'get_market_prices', 'web_search', 'visuals_core'

#### Scenario: Selected tools execute correctly
- **WHEN** user creates a task with selected tools
- **AND** task is executed
- **THEN** the selected tool names correctly match registry entries
- **AND** tools execute without errors due to name mismatch