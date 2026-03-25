## Context

This change improves the web portal task management UI and fixes a tool name mismatch issue. The current implementation uses full-page navigation with raw YAML editing, and the TaskEditor fetches tool names from `/api/skills` which returns skill file names rather than actual tool registry names.

Additionally, this change addresses multiple UI consistency issues across the portal:
- Config dashboard uses raw HTML buttons instead of shadcn/ui Button components
- Dashboard titles have inconsistent styling (different fonts, icons, spacing)
- "Mood" label should be "Status" for clarity
- Robot avatar placeholder is not centered
- Uptime counter stops after initial API fetch
- Config dashboard uses emoji instead of Lucide icons

## Goals / Non-Goals

**Goals:**
- Improve task management UX with split-view interface
- Replace raw YAML editing with structured form fields
- Fix tool name mismatch by exposing actual tool names from registry
- Standardize all UI components to use shadcn/ui Button components
- Ensure consistent dashboard title styling across all pages
- Fix minor UI bugs (avatar centering, label naming, uptime counter)

**Non-Goals:**
- Change backend task file storage format (YAML remains unchanged)
- Modify task scheduler execution logic
- Add new tool capabilities (just exposing existing tools)
- Redesign the overall portal layout or color scheme

## Decisions

1. **Split-view Layout**: Use CSS flexbox with two columns instead of conditional rendering
   - Rationale: Provides persistent access to task list while editing
   - Alternative considered: Modal overlay - rejected because it blocks visibility of list

2. **Client-side form validation**: Validate cron expressions and required fields in browser
   - Rationale: Faster feedback than server round-trip
   - Alternative considered: Server validation - would require additional API calls

3. **New `/api/tools` endpoint**: Create separate endpoint instead of modifying `/api/skills`
   - Rationale: Maintains backward compatibility, separates concerns (skills are files, tools are registry entries)
   - Alternative considered: Modify existing `/api/skills` - would break existing consumers

4. **Tool registry integration**: Fetch tool names from registry.py (get_openai_tools keys)
   - Rationale: Returns actual callable tool names that match the execution layer
   - Alternative considered: Parse skill file names - would still have mismatch with registry

5. **shadcn/ui Button Components**: Replace all raw HTML buttons with shadcn/ui Button components
   - Rationale: Consistent styling, proper accessibility, easier theming
   - Alternative considered: Keep inline styles - rejected for inconsistency

6. **Lucide Icons**: Use Lucide icons from `lucide-react` package (already installed)
   - Rationale: Consistent with existing codebase, better accessibility, scalable
   - Alternative considered: Keep emoji - rejected for inconsistency with UI style

7. **Local Uptime Counter**: Implement client-side counter that increments every minute
   - Rationale: Provides real-time feedback without excessive API calls
   - Alternative considered: Poll API every minute - would increase server load

8. **Unsaved Changes Detection**: Track form state changes in TaskEditor
   - Rationale: Prevent accidental data loss when switching tasks
   - Alternative considered: Auto-save - would require more complex backend integration

## Risks / Trade-offs

- [Risk] Form fields may not cover all YAML configurations → Mitigation: Include YAML view toggle for advanced users
- [Risk] New endpoint increases API surface → Mitigation: Simple response format, no new dependencies
- [Risk] Mobile view complexity with split panels → Mitigation: Full-screen overlay on mobile
- [Risk] Button style changes may affect user muscle memory → Mitigation: Keep button labels and positions consistent
- [Risk] Local uptime counter may drift from actual server uptime → Mitigation: Refresh from API periodically (every 5 minutes)

## Open Questions

- Should the tool dropdown include tool descriptions for better UX?
- Should we preserve the raw YAML view as the default or as a toggle?
- Should the unsaved changes dialog also apply to PersonaEditor?
- How often should we refresh the uptime from the API to stay in sync?