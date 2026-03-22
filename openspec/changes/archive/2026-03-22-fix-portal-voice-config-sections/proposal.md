## Why

The Web Portal's ConfigEditor doesn't display the Portal and Voice configuration sections, even though they're defined in CONFIG_SECTIONS. Users cannot edit these settings through the UI. This is due to mismatched field keys between the backend API response and frontend field lookup logic.

## What Changes

1. **Backend - config.py**:
   - Add `voice.key` and `voice.region` to `EDITABLE_FIELDS` for voice configuration
   - Update `read_only_fields` to use `voice` instead of `azure-speech` for consistency

2. **Frontend - ConfigEditor.tsx**:
   - Add nested portal fields (`portal.enabled`, `portal.port`, `portal.logs.retention_days`, `portal.logs.levels`) to CONFIG_SECTIONS
   - Add nested voice fields (`voice.key`, `voice.region`, `voice.default_voice`) to CONFIG_SECTIONS
   - Fix field extraction to look for portal data in `apiData.portal` (separate field) instead of `apiData.config.portal`
   - Fix field extraction to look for voice data in `apiData.config.voice` with proper nested key matching

## Capabilities

### New Capabilities
None - this is a bug fix to existing functionality.

### Modified Capabilities
- `web-portal`: Fix ConfigEditor to properly display and edit Portal and Voice configuration sections

## Impact

- **Files Modified**:
  - `bot/web/routes/config.py` - Add voice fields to EDITABLE_FIELDS
  - `web/src/components/ConfigEditor.tsx` - Fix CONFIG_SECTIONS and field extraction logic

- **API Changes**: None (existing API already returns portal and voice data, just in different structure)
- **Breaking Changes**: None