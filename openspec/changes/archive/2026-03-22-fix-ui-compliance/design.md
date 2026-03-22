## Context

The ConfigEditor component (`web/src/components/ConfigEditor.tsx`) was built before the formal UI standards in `openspec/specs/web-portal/UI_CONVENTIONS.md` were completed. Now that the spec is formalized, the implementation needs to be aligned with the standards.

Current state vs spec:
- Buttons use default gray (#1a1a1a) instead of green/blue/red per spec
- Sidebar is 160px instead of 200px
- Input fields lack exact styling (height, colors)
- No loading opacity on buttons during save operations

## Goals / Non-Goals

**Goals:**
- Align ConfigEditor.tsx with all UI_CONVENTIONS.md standards
- Apply exact button colors (Save=#22c55e, Apply=#3b82f6, Save&Apply=#22c55e)
- Add loading state opacity (0.5) when saving
- Fix sidebar width to 200px
- Apply proper input field styling
- Style read-only fields appropriately

**Non-Goals:**
- No new functionality or features
- No API changes
- No behavior changes - purely visual alignment
- Not fixing TaskEditor or PersonaEditor (separate work if needed)

## Decisions

### D1: Inline Styles vs CSS Classes
**Decision:** Use inline styles with consistent values from spec  
**Rationale:** The existing codebase uses inline styles extensively. Adding CSS classes would require refactoring multiple components. Inline styles with spec-consistent values maintain consistency.

**Alternative Considered:** Create shared CSS classes
- Rejected: Would require modifying index.css and adding className to many elements

### D2: Button Styling Strategy (Updated March 2026)
**Decision:** Apply dark-theme-optimized colors  
**Rationale:** Bright colors (#22c55e, #3b82f6) were too vivid for dark backgrounds. Updated to more muted tones per UI_CONVENTIONS Section 2.

| Button | Old Color | New Color | When |
|--------|-----------|-----------|------|
| Save | #22c55e | #10b981 | Primary action |
| Apply | #3b82f6 | #6366f1 | Reload in-memory |
| Save&Apply | #22c55e | #10b981 | Save + reload |
| Delete | #dc2626 | #ef4444 | Danger zone |
| Cancel | #4b5563 | #4b5563 | Cancel (unchanged) |

### D2b: Button Centering
**Decision:** Add flex centering to all buttons  
**Rationale:** Text was not vertically centered in buttons. Added `display: 'flex', alignItems: 'center', justifyContent: 'center'`.

### D2c: Icon Visibility Fix
**Decision:** Add explicit stroke styling to Heroicons  
**Rationale:** Outline icons use `stroke="currentColor"` which renders too faintly on dark backgrounds. Added explicit `strokeWidth: 2` and matching color.

### D3: Loading State Implementation
**Decision:** Use conditional opacity and text change  
**Rationale:** Per spec Section 19, buttons should:
- Have `opacity: 0.5` when saving
- Show "Saving..." text during operation
- Be disabled to prevent double-clicks

## Risks / Trade-offs

- **[Risk] CSS conflicts** → Using inline styles avoids conflicts with index.css
- **[Risk] Inconsistent with other components** → TaskEditor/PersonaEditor will need similar fixes later (documented as non-goal)
- **[Trade-off] Longer component file** → Inline styles add lines but maintain consistency with existing pattern

## Migration Plan

1. Update ConfigEditor.tsx button styles to match spec colors
2. Add opacity: 0.5 to buttons when saving state is true
3. Change sidebar width from 160px to 200px
4. Apply exact input field styling per spec
5. Add read-only field styling (opacity 0.7, background #151515)
6. Verify no regressions with manual testing

## Open Questions

- None - the spec is clear on required changes