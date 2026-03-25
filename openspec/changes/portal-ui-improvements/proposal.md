## Why

The current task management interface in the web portal uses a full-page navigation pattern with raw YAML editing, which is cumbersome for users. Additionally, the TaskEditor tool dropdown shows skill file names (e.g., "yahoo_finance") instead of actual tool names (e.g., "get_market_prices"), creating confusion when users select tools that don't match the registry. 

Beyond the task management issues, the web portal has several UI consistency problems: the Config dashboard uses inconsistent button styles (raw HTML buttons instead of shadcn/ui Button components), the "Mood" label should be "Status", the robot avatar placeholder is not centered, the uptime counter stops after initial load, and dashboard titles have inconsistent styling. These UI improvements will enhance usability, fix the tool name mismatch, and establish consistent UI patterns using shadcn/ui components throughout the portal.

## What Changes

1. **Task Split-View Interface**: Convert task management from full-page navigation to a split-view panel layout where TaskList and TaskEditor are always visible side by side
2. **Structured Form Fields**: Replace raw YAML textarea with structured form fields (Name, Cron, Enabled, Model, Persona, Tools, User ID, Channel ID, Prompt)
3. **Tool Name Alignment**: Create new `/api/tools` endpoint to return actual tool names from the registry, and update TaskEditor to use it
4. **Responsive Design**: Add responsive breakpoints for mobile (stacked), tablet, and desktop (split-view)
5. **Unsaved Changes Protection**: Add confirmation dialog when clicking "Add New" while editing an unsaved task
6. **Config Dashboard Button Styling**: Replace raw HTML buttons with shadcn/ui Button components for Save, Apply, and Save&Apply buttons
7. **Uptime Counter**: Implement local counter that continues incrementing after receiving initial data from API
8. **Avatar Centering Fix**: Fix robot emoji centering in avatar placeholder (justify-content → justify-center)
9. **Label Rename**: Change "Mood" to "Status" in dashboard
10. **Emoji to Icons**: Replace emoji icons with Lucide icons in Config dashboard section navigation
11. **Dashboard Title Consistency**: Standardize all dashboard titles to use consistent styling with icons

## Capabilities

### New Capabilities
- `task-split-view`: Split-view interface for task management with structured form fields
- `tool-name-api`: New API endpoint `/api/tools` that returns tool names and schemas from the registry
- `unsaved-changes-dialog`: Confirmation dialog when navigating away from unsaved task edits
- `live-uptime-counter`: Real-time uptime counter that continues counting after initial data load

### Modified Capabilities
- `task-management`: UI requirements changing from full-page navigation to split-view panel, and from raw YAML editing to structured form fields
- `portal-ui-consistency`: Unified UI styling across all dashboards using shadcn/ui components

## UI Style Guidelines (shadcn/ui)

All UI improvements MUST follow the shadcn/ui design system:
- Use shadcn/ui `Button` component for all interactive buttons
- Use shadcn/ui `Input`, `Select`, `Checkbox`, `Dialog` for form elements
- Use Lucide icons (from `lucide-react`) instead of emoji
- Follow consistent color tokens: `primary`, `secondary`, `outline`, `destructive`, `ghost`
- Use Tailwind CSS utility classes for layout and spacing

### Reference Components
- Button variants: `default`, `secondary`, `outline`, `ghost`, `destructive`
- Common patterns: `<Button variant="outline" size="sm">`
- Icons: Import from `lucide-react` (e.g., `import { Save, RotateCcw, Plus } from 'lucide-react'`)

## Impact

- **Frontend**: Changes to Dashboard.tsx, TaskList.tsx, TaskEditor.tsx, ConfigEditor.tsx, ServerList.tsx, PersonaList.tsx, SkillsList.tsx
- **Backend**: New endpoint in bot/web/routes/ (or bot/web/server.py)
- **API**: New `/api/tools` endpoint alongside existing `/api/skills`
- **Dependencies**: No new external dependencies (uses existing shadcn/ui and lucide-react)