## Why

The current ConfigEditor implementation (`web/src/components/ConfigEditor.tsx`) does not follow the UI standards defined in `openspec/specs/web-portal/UI_CONVENTIONS.md`. This creates visual inconsistency and violates the design contract. The implementation was built before the spec was formalized, and now needs to be aligned.

## What Changes

- Fix button colors to match spec (Save=#22c55e green, Apply=#3b82f6 blue, Delete=#dc2626 red)
- Add button dimensions (36px height, 80px min-width, 12px gap)
- Add loading state opacity (0.5 when saving) to buttons
- Fix sidebar width from 160px to 200px
- Apply exact input field styling (40px height, #1a1a1a background, #404040 border)
- Add proper styling for read-only fields (opacity 0.7, #151515 background)
- Align all other UI elements with formal spec

## Capabilities

### New Capabilities
<!-- No new capabilities - this is an implementation alignment fix -->

### Modified Capabilities
- `web-portal/ui-conventions`: Implementation alignment (no requirement changes, just compliance)

## Impact

- **Files Modified**: 
  - `web/src/components/ConfigEditor.tsx` (main changes)
  - `web/src/components/FormModal.tsx` (button alignment)
  - `web/src/index.css` (if additional styles needed)
- **No breaking changes**: All existing functionality preserved
- **Visual only**: No API or behavior changes