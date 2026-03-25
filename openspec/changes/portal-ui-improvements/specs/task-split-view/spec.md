# Task Split-View Interface

## ADDED Requirements

### Requirement: Split-view layout for task management
The task management interface SHALL use a split-view layout with TaskList always visible alongside TaskEditor.

#### Scenario: Split-view on desktop
- **WHEN** user navigates to Tasks tab on desktop (≥1024px)
- **THEN** TaskList is displayed on the left (50% width) and TaskEditor on the right (50% width)
- **AND** both panels are always rendered

#### Scenario: Split-view on tablet
- **WHEN** user navigates to Tasks tab on tablet (768-1023px)
- **THEN** TaskList is displayed on the left (40% width) and TaskEditor on the right (60% width)

#### Scenario: Stacked layout on mobile
- **WHEN** user navigates to Tasks tab on mobile (<768px)
- **THEN** TaskList is shown first with a selected task opening as a full-screen overlay
- **AND** a back button returns to the list view

### Requirement: Structured form fields for task editing
The TaskEditor SHALL provide structured form fields instead of raw YAML editing.

#### Scenario: Create new task with form fields
- **WHEN** user clicks "Add New" button
- **THEN** form fields are displayed: Name, Cron, Enabled, Model, Persona, Prompt, Tools, User ID, Channel ID
- **AND** validation errors are shown inline for invalid input

#### Scenario: Edit existing task
- **WHEN** user selects an existing task
- **THEN** the task data is loaded into the form fields
- **AND** changes can be saved or cancelled

### Requirement: Form field validation
The form SHALL validate input before allowing save.

#### Scenario: Invalid cron expression
- **WHEN** user enters an invalid cron expression
- **THEN** an error message is displayed below the Cron field
- **AND** save button is disabled until corrected

#### Scenario: Required field empty
- **WHEN** user attempts to save without filling required fields (Name, Prompt)
- **THEN** error messages are displayed for missing required fields
- **AND** save is blocked until corrected

## MODIFIED Requirements

### Requirement: List existing tasks
The system SHALL provide an endpoint to list all available tasks with their full content.

#### Scenario: List tasks
- **WHEN** user requests GET /api/tasks
- **THEN** response includes array of {name, content, enabled, status} for each task
- **AND** UI displays tasks in a card list format with visual selection indicator

#### Scenario: Select task from list
- **WHEN** user clicks on a task card
- **THEN** the selected card shows a visual indicator (border highlight)
- **AND** TaskEditor populates with the task data