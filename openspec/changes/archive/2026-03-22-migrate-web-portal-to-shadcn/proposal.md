## Why

The current web portal uses inline React styles and inconsistent component patterns. This creates maintainability issues, makes theming difficult, and lacks accessibility best practices. Migrating to shadcn/ui will provide a consistent, accessible, and maintainable component library while preserving all existing functionality.

## What Changes

- Migrate all UI components to shadcn/ui component library
- Replace inline React styles with Tailwind CSS classes
- Extract business logic into custom React hooks for better separation of concerns
- Update component file structure to follow shadcn conventions
- Fix the existing button component (currently uses @base-ui/react instead of standard shadcn)
- Ensure dark theme consistency across all components
- Add proper TypeScript types to all components

## Capabilities

### New Capabilities

- **component-migration**: Comprehensive migration of all web portal components to shadcn/ui
- **hook-extraction**: Extract reusable business logic into custom React hooks (useConfig, useProviders, useModels, usePersonas, useTasks, useServers)
- **style-standardization**: Replace all inline styles with Tailwind CSS classes following UI_CONVENTIONS

### Modified Capabilities

- **web-portal**: Update the web-portal spec to reference shadcn/ui as the component library and add component-specific requirements

## Impact

### Code Impact

**Components to migrate:**
- `ConfigEditor.tsx` (~900 lines) - Main config form with tabs, tables, modals
- `Sidebar.tsx` - Navigation sidebar
- `TaskList.tsx` + `TaskEditor.tsx` (~400 lines combined)
- `PersonaList.tsx` + `PersonaEditor.tsx` (~300 lines combined)
- `ServerList.tsx` + `ServerDrawer.tsx` + `ServerDetail.tsx` (~300 lines)
- `LogViewer.tsx` - Log display
- `Login.tsx` (~100 lines) - Authentication form
- `FormModal.tsx` + `Modal.tsx` - Reusable dialogs
- `Dashboard.tsx` - Main dashboard

**New components to add (shadcn):**
- Button, Input, Checkbox, Select, Label
- Table, Tabs, Dialog, Card, Form
- ScrollArea, Separator, Badge

**New hooks to create:**
- `useConfig` - Config fetching and management
- `useProviders` - LLM provider CRUD
- `useModels` - LLM model CRUD
- `usePersonas` - Persona management
- `useTasks` - Task management
- `useServers` - Server listing

### Dependencies

- shadcn/ui already partially configured in project
- Need to add: dialog, table, tabs, select, input, checkbox, label, card, form, scroll-area, separator, badge components

**Critical: @radix-ui primitives required**
- @radix-ui/react-slot (for Button component - replacing @base-ui/react)
- @radix-ui/react-dialog (for Dialog component)
- @radix-ui/react-tabs (for Tabs component)
- @radix-ui/react-select (for Select component)
- @radix-ui/react-checkbox (for Checkbox component)
- @radix-ui/react-label (for Label component)
- @radix-ui/react-scroll-area (for ScrollArea component)
- @radix-ui/react-separator (for Separator component)
- @radix-ui/react-toggle (for Toggle/Tabs components)

**Dependencies to remove after migration:**
- @base-ui/react (currently used by broken button component)

### Systems

- Web portal frontend only (no backend API changes)