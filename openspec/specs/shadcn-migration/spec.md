# shadcn-migration Specification

## Purpose
Document the migration of the web portal from custom inline-styled components to shadcn/ui component library. This includes component replacements, dark theme standards, and icon library consolidation.
## Requirements
### Requirement: Web portal uses shadcn/ui component library
The web portal SHALL use shadcn/ui components for all UI elements, replacing custom inline-styled components.

#### Scenario: Components replaced with shadcn equivalents
- **WHEN** a component is migrated to shadcn
- **THEN** the component uses shadcn components (Button, Input, Dialog, Table, Tabs, etc.)
- **AND** all inline React styles are replaced with Tailwind CSS classes
- **AND** lucide-react icons are used instead of heroicons

### Requirement: Button component uses standard shadcn
The portal SHALL use the standard shadcn Button component instead of @base-ui/react.

#### Scenario: Button component fixed
- **WHEN** the button component is refactored
- **THEN** it uses @radix-ui/react-slot as the primitive
- **AND** supports variants: default, outline, secondary, ghost, destructive, link
- **AND** supports sizes: default, sm, lg, icon

### Requirement: Business logic extracted to custom hooks
The portal SHALL extract business logic into reusable React hooks.

#### Scenario: Config hook extracted
- **WHEN** config management is needed
- **THEN** useConfig() hook provides: loading state, config data, save function, error handling

#### Scenario: Provider hook extracted
- **WHEN** LLM provider management is needed
- **THEN** useProviders() hook provides: list, add, update, delete operations

#### Scenario: Model hook extracted
- **WHEN** LLM model management is needed
- **THEN** useModels() hook provides: list, add, update, delete operations

### Requirement: Consistent dark theme across components
The portal SHALL maintain consistent dark theme styling using Tailwind CSS.

#### Scenario: Dark theme colors
- **WHEN** any component is rendered
- **THEN** it uses semantic Tailwind colors (bg-background, text-foreground, etc.)
- **AND** uses dark theme background: #1a1a1a
- **AND** uses dark theme borders: #404040

#### Scenario: Form field styling
- **WHEN** any input field is rendered
- **THEN** it uses background: #1a1a1a
- **AND** border: 1px solid #404040
- **AND** focus border: #404040

#### Scenario: Number input spinners hidden
- **WHEN** a number input is rendered
- **THEN** spinner buttons are hidden via CSS

### Requirement: ConfigEditor migrated to shadcn
The ConfigEditor component SHALL be fully migrated to shadcn/ui.

#### Scenario: Config sections use Tabs
- **WHEN** user views ConfigEditor
- **THEN** section navigation uses shadcn Tabs component

#### Scenario: LLM providers table uses Table component
- **WHEN** user views Providers tab
- **THEN** the table renders using shadcn Table components

#### Scenario: Provider/Model modals use Dialog
- **WHEN** user adds or edits a provider/model
- **THEN** the modal uses shadcn Dialog component

#### Scenario: API key field masking
- **WHEN** provider API key is displayed
- **THEN** default shows masked "••••••••"
- **AND** Eye icon toggle button switches between password/text type

#### Scenario: Form inputs use Input/Select components
- **WHEN** user fills out any form field
- **THEN** the field uses shadcn Input, Select, or Checkbox components

### Requirement: Sidebar uses shadcn components
The Sidebar navigation SHALL use shadcn Button components.

#### Scenario: Sidebar navigation items
- **WHEN** user views the sidebar
- **THEN** each navigation item uses Button with appropriate variant

### Requirement: Task components use shadcn
TaskList and TaskEditor SHALL use shadcn components.

#### Scenario: Task list displays in table
- **WHEN** user views task list
- **THEN** tasks display in a shadcn Table

#### Scenario: Task editor in dialog
- **WHEN** user adds or edits a task
- **THEN** editor appears in a shadcn Dialog

### Requirement: Persona components use shadcn
PersonaList and PersonaEditor SHALL use shadcn components.

#### Scenario: Persona list displays in table
- **WHEN** user views persona list
- **THEN** personas display in a shadcn Table

#### Scenario: Persona editor in dialog
- **WHEN** user adds or edits a persona
- **THEN** editor appears in a shadcn Dialog

### Requirement: Server components use shadcn
ServerList, ServerDetail, and ServerDrawer SHALL use shadcn components.

#### Scenario: Server list in cards
- **WHEN** user views server list
- **THEN** servers display in shadcn Card components

#### Scenario: Server detail in dialog
- **WHEN** user views server details
- **THEN** details display in a shadcn Dialog or Drawer

### Requirement: Login form uses shadcn
The Login component SHALL use shadcn form components.

#### Scenario: Login form fields
- **WHEN** user views login page
- **THEN** username and password fields use shadcn Input
- **AND** submit uses shadcn Button with loading state

### Requirement: Icon library unified to lucide-react
The portal SHALL use lucide-react for all icons.

#### Scenario: Icon replacement
- **WHEN** any component uses an icon
- **THEN** lucide-react icons are used (not heroicons)
- **AND** icons match shadcn sizing conventions

### Requirement: Build verification after foundation setup
The project SHALL pass TypeScript compilation after shadcn components are installed.

#### Scenario: Build passes
- **WHEN** `npm run build` is executed in the web directory
- **THEN** no TypeScript errors are present
- **AND** the application bundles successfully

### Requirement: Responsive layout
The portal SHALL provide responsive layouts for different screen sizes.

#### Scenario: Desktop layout (>1024px)
- **WHEN** user views portal on desktop
- **THEN** sidebar is 200px fixed
- **AND** content area is flexible with max 800px

#### Scenario: Tablet layout (768px-1024px)
- **WHEN** user views portal on tablet
- **THEN** sidebar collapses to 180px or becomes collapsible

#### Scenario: Mobile layout (<768px)
- **WHEN** user views portal on mobile
- **THEN** sidebar is hidden
- **AND** hamburger menu provides navigation access
- **AND** content uses full width with stacked layout

