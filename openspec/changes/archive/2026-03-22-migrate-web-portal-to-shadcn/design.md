## Context

### Current State

The web portal is a React application built with:
- Vite as the build tool
- TypeScript for type safety
- Tailwind CSS for some styling (but many inline styles)
- Heroicons for icons
- react-hot-toast for notifications
- Custom component architecture

**Existing components:**
- ConfigEditor (~900 lines) - Complex form with tabs, tables, modals
- Sidebar - Navigation
- TaskList/TaskEditor (~400 lines)
- PersonaList/PersonaEditor (~300 lines)
- ServerList/ServerDrawer/ServerDetail (~300 lines)
- LogViewer, Login, FormModal, Modal, Dashboard

**Current issues:**
- Button component uses @base-ui/react (non-standard)
- Heavy use of inline React styles (inputStyle, buttonStyle objects)
- No clear separation between UI and business logic
- Inconsistent component patterns across files

### Constraints

1. Preserve all existing functionality - no breaking changes to API behavior
2. Maintain dark theme as primary (matching UI_CONVENTIONS)
3. Keep using Tailwind CSS (already partially configured)
4. Continue using react-hot-toast for notifications
5. Use lucide-react for icons (as per shadcn config)

### Stakeholders

- Users of the web portal (Discord bot administrators)
- Developers maintaining the codebase

### Current Project State (Verified)

**Already configured:**
- `components.json` exists with base-vega style, lucide icons
- `tailwind.config.js` exists
- `index.css` has shadcn imports
- `lib/utils.ts` has cn() function
- `lucide-react@0.577.0` installed

**Issues to fix:**
- `button.tsx` uses `@base-ui/react` (NOT standard shadcn) - needs rewrite
- No UI components installed (only button.tsx exists)
- `@base-ui/react` present in package.json (to be removed)
- `heroicons` present (to be replaced with lucide)

**Dependencies already installed:**
- class-variance-authority, clsx, tailwind-merge
- tw-animate-css

## Goals / Non-Goals

**Goals:**
1. Migrate all components to shadcn/ui library
2. Replace inline styles with Tailwind CSS classes
3. Extract business logic into custom React hooks
4. Fix the broken button component (@base-ui/react → standard shadcn)
5. Achieve consistent dark theme across all components
6. Improve TypeScript type coverage

**Non-Goals:**
- No backend API changes
- No routing changes
- No authentication flow changes
- No new functionality - purely UI refactoring
- No database schema changes

## Decisions

### Decision 1: Use shadcn/ui as component library

**Rationale:** shadcn/ui provides accessible, customizable components that integrate well with Tailwind CSS. It follows best practices for React component design.

**Alternative considered:** Use raw Radix UI primitives
- Radix provides unstyled accessible components, but requires more setup
- shadcn provides pre-styled components that match our UI_CONVENTIONS better

### Decision 2: Extract hooks for business logic

**Rationale:** Currently, each component mixes UI rendering with API calls and state management. Extracting hooks provides:
- Better separation of concerns
- Reusability across components
- Easier testing

**Hook structure:**
- `useConfig()` - Fetch/save configuration
- `useProviders()` - LLM provider CRUD
- `useModels()` - LLM model CRUD
- `usePersonas()` - Persona management
- `useTasks()` - Task management
- `useServers()` - Server listing

### Decision 3: Phased migration approach

**Rationale:** Migrating all components at once is risky. Phased approach allows:
- Incremental testing
- Easy rollback of individual components
- Learning from early migrations

**Migration order:**
1. Foundation: Fix button component, add missing shadcn components
2. Simple components: Login, Modal, Sidebar
3. Medium complexity: TaskList, PersonaList, ServerList
4. Complex: ConfigEditor (largest, most interdependent)

### Decision 4: Preserve react-hot-toast for notifications

**Rationale:** react-hot-toast is already well-integrated and matches UI_CONVENTIONS for toast notifications. shadcn's sonner could be an alternative, but migration would add complexity without significant benefit.

**Alternative:** Migrate to sonner (shadcn's toast)
- Would require changing all toast calls across components
- No functional advantage over react-hot-toast

### Decision 5: Switch from Heroicons to Lucide React

**Rationale:** 
- shadcn/ui uses Lucide by default
- Already installed in package.json (lucide-react@0.577.0)
- Better consistency with shadcn components

**Alternative:** Keep Heroicons
- Would require custom icon props on shadcn components
- Inconsistent with shadcn documentation and examples

## Migration Plan

### Phase 1: Foundation (Quick Wins)
1. Fix button component - replace @base-ui/react with standard shadcn Button
2. Add required shadcn components: Input, Checkbox, Label, Card
3. Update vite.config.ts to resolve @base-ui paths if needed

### Phase 2: Simple Component Migration
1. Login.tsx → Input, Button, Label
2. Modal.tsx + FormModal.tsx → Dialog
3. Sidebar → NavigationMenu or custom with shadcn Button

### Phase 3: List Components
1. TaskList → Table, Button, Dialog
2. PersonaList → Table, Button, Dialog
3. ServerList → Table, Card, Button
4. LogViewer → ScrollArea, Badge

### Phase 4: Complex Component Migration
1. ConfigEditor → Tabs, Table, Dialog, Select, Input, Checkbox
2. Extract hooks: useConfig, useProviders, useModels

### Phase 5: Cleanup
1. Remove unused inline styles
2. Remove Heroicons imports (replace with Lucide)
3. Add custom hooks to hooks/ directory
4. Run full integration tests

## Risks / Trade-offs

### Risk 1: Component behavior differences
**Risk:** shadcn components may behave differently than current implementation
**Mitigation:** Test each component thoroughly after migration; preserve all existing props and handlers

### Risk 2: Icon migration complexity
**Risk:** Replacing Heroicons with Lucide requires updating all icon imports
**Mitigation:** Use shadcn's icon mapping; Lucide has similar icons to Heroicons with minor name differences

### Risk 3: Type compatibility
**Risk:** Custom button component may have different props than shadcn Button
**Mitigation:** Review and update component props; may need wrapper components

### Risk 4: Breaking existing functionality
**Risk:** Changes might inadvertently break existing portal features
**Mitigation:** Test after each component migration; maintain API compatibility (no backend changes)

### Trade-off: Time vs. Quality
- Full migration will take longer than quick fixes
- But provides long-term maintainability benefits
- Better for onboarding new developers

## Open Questions

1. **Should we keep @base-ui/react for anything?** - Currently only used in button. Likely can be removed entirely.

2. **How to handle the JSON edit mode in ConfigEditor?** - This is a unique feature. Should we keep it as-is or migrate to a shadcn code editor component?

3. **Should we use shadcn's Form component with react-hook-form?** - Currently using uncontrolled forms. Adding react-hook-form would add dependencies but improve validation. Decision: Keep uncontrolled for now, revisit if needed.

4. **Mobile responsiveness** - Current sidebar collapses to icons. Should we use shadcn's Collapsible or Sheet for mobile navigation? Decision: Keep current behavior, enhance with shadcn components where helpful.