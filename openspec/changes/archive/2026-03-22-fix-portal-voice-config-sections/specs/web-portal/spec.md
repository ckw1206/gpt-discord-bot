## Purpose

Provide a web-based admin portal for managing the Discord bot.

## ADDED Requirements

### Requirement: Portal config section editable in ConfigEditor
The system SHALL allow editing of portal configuration fields (enabled, port, cors_origins, logs) through the ConfigEditor UI.

#### Scenario: Portal section displays in ConfigEditor sidebar
- **WHEN** user navigates to Config tab
- **THEN** Portal section is visible in the sidebar navigation

#### Scenario: Portal enabled toggle works
- **WHEN** user toggles portal.enabled from true to false
- **AND** clicks Save
- **THEN** config.yaml is updated with portal.enabled: false

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