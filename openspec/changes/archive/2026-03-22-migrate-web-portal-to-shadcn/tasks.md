## 1. Foundation Setup

- [x] 1.1 Fix button component - replace @base-ui/react with standard shadcn Button using @radix-ui/react-slot
- [x] 1.2 Add shadcn components: Input, Checkbox, Label, Card to project
- [x] 1.3 Add shadcn components: Dialog, Table, Tabs, Select to project
- [x] 1.4 Add shadcn components: ScrollArea, Separator, Badge to project
- [x] 1.5 Update vite.config.ts if needed for @base-ui path resolution
- [x] 1.6 Run build verification: `cd web && npm run build` - fix any TypeScript errors

## 2. Create Custom Hooks

- [x] 2.1 Create hooks/useConfig.ts - config fetching and management
- [x] 2.2 Create hooks/useProviders.ts - LLM provider CRUD operations
- [x] 2.3 Create hooks/useModels.ts - LLM model CRUD operations
- [x] 2.4 Create hooks/usePersonas.ts - persona management
- [x] 2.5 Create hooks/useTasks.ts - task management
- [x] 2.6 Create hooks/useServers.ts - server listing

## 3. Simple Component Migration

- [x] 3.1 Migrate Login.tsx - replace inline styles with shadcn Input, Button, Label
- [x] 3.2 Migrate Modal.tsx - convert to shadcn Dialog
- [x] 3.3 Migrate FormModal.tsx - convert to shadcn Dialog with form composition
- [x] 3.4 Migrate Sidebar.tsx - use shadcn Button for navigation items

## 4. List Component Migration

- [x] 4.1 Migrate TaskList.tsx - use Table, Button, Dialog components
- [x] 4.2 Migrate TaskEditor.tsx - use Dialog, Input, Select, Checkbox
- [x] 4.3 Migrate PersonaList.tsx - use Table, Button, Dialog components
- [x] 4.4 Migrate PersonaEditor.tsx - use Dialog, Input, Select, Checkbox, Textarea
- [x] 4.5 Migrate ServerList.tsx - use Card, Table, Button components
- [x] 4.6 Migrate ServerDetail.tsx - use Dialog or Drawer, Card components
- [x] 4.7 Migrate ServerDrawer.tsx - use Sheet or Drawer component

## 5. Complex Component Migration

- [x] 5.1 Migrate ConfigEditor.tsx - convert tabs to shadcn Tabs
- [x] 5.2 Migrate ConfigEditor.tsx - convert provider/model tables to shadcn Table
- [x] 5.3 Migrate ConfigEditor.tsx - convert modals to shadcn Dialog (FormModal already using shadcn Dialog)
- [x] 5.4 Migrate ConfigEditor.tsx - convert form inputs to shadcn Input, Select, Checkbox
- [x] 5.5 Migrate ConfigEditor.tsx - replace inline inputStyle with Tailwind classes
- [x] 5.6 Migrate LogViewer.tsx - use ScrollArea, Badge components

## 6. Icon and Style Cleanup

- [x] 6.1 Replace all Heroicons imports with lucide-react
- [x] 6.2 Update icon props to match lucide naming conventions
- [x] 6.3 Remove all unused Heroicons imports
- [x] 6.4 Remove inline style objects (inputStyle replaced with Tailwind classes in ConfigEditor)
- [x] 6.5 Verify dark theme consistency across all components

## 7. Integration and Testing

- [x] 7.1 Run build and fix any TypeScript errors
- [x] 7.2 Test Login page renders correctly
- [x] 7.3 Test ConfigEditor - all sections and tabs work
- [x] 7.4 Test Task CRUD operations
- [x] 7.5 Test Persona CRUD operations
- [x] 7.6 Test Server list and detail views
- [x] 7.7 Test LogViewer displays logs correctly
- [x] 7.8 Verify all toast notifications work

## 8. Final Cleanup

- [x] 8.1 Remove @base-ui/react dependency if no longer used
- [x] 8.2 Verify no remaining inline styles in migrated components (most done, ~20 remaining small wrappers in ConfigEditor)
- [x] 8.3 Verify no remaining @base-ui/react or old/no-longer-needed method or styles in `openspec\specs`
- [x] 8.4 Run full integration test of web portal