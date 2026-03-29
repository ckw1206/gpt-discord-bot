## Context

The ConfigEditor component in the web portal has two configuration sections that aren't displaying:
- **Portal** section (configured in `config.yaml` under `portal:`)
- **Voice** section (configured in `config.yaml` under `voice:`)

### Current Backend Behavior (bot/web/routes/config.py)

1. **Portal Config**: Returned separately via `_get_portal_config_safe()` as `apiData.portal`:
   ```python
   {
     "enabled": bool,
     "port": int,
     "logs": { "retention_days": int, "levels": list },
     "require_discord_admin": bool
   }
   ```

2. **Voice Config**: Part of the main config under `config.voice`:
   ```python
   {
     "key": str,
     "region": str,
     "default_voice": str
   }
   ```

3. **EDITABLE_FIELDS** (current):
   ```python
   EDITABLE_FIELDS = {
     "discord.status_message", "discord.client_id", "discord.bot_token",
     "behavior.max_text", "behavior.max_images", "behavior.max_messages",
     "behavior.use_plain_responses", "behavior.show_embed_color", "behavior.allow_dms",
     "llm.persona", "llm.system_prompt", "llm.fallback_models",
     "portal",  # <-- Only top-level key exists
   }
   ```
   Note: `voice` is NOT in EDITABLE_FIELDS.

4. **read_only_fields** (current):
   ```python
   ["providers", "models", "fallback_models", "tools", "azure-speech", "permissions"]
   ```
   Note: Uses legacy `azure-speech` key which doesn't match current config structure.

### Current Frontend Behavior (ConfigEditor.tsx)

1. **CONFIG_SECTIONS** (current):
   ```typescript
   const CONFIG_SECTIONS = [
     { id: 'discord', fields: ['discord.status_message', ...] },
     { id: 'behavior', fields: ['behavior.max_text', ...] },
     { id: 'llm', fields: ['llm.providers', ...] },
     { id: 'voice', fields: ['voice', 'azure-speech'] },  // Doesn't match API keys
     { id: 'portal', fields: ['portal'] },  // Only top-level, doesn't match nested API keys
   ]
   ```

2. **Field Extraction Logic**:
   ```typescript
   // Looks for value in apiData.config
   let value = getNestedValue(apiData.config, key)
   ```
   - For portal: looks for `apiData.config.portal` but API returns `apiData.portal`
   - For voice: key doesn't match (voice not in editable_fields, uses azure-speech in read_only)

## Goals / Non-Goals

**Goals:**
1. Make Portal section display in ConfigEditor with nested fields editable
2. Make Voice section display in ConfigEditor with nested fields editable
3. Ensure changes can be saved and persisted to config.yaml
4. Maintain consistency with existing ConfigEditor patterns

**Non-Goals:**
1. Adding completely new portal/voice configuration fields (only fixing display)
2. Hot-reload functionality for config changes (requires separate feature)
3. Adding validation for portal/voice fields (existing validation covers this)

## Decisions

### Decision 1: How to handle portal config structure?

**Option A**: Frontend looks in separate `apiData.portal` field
**Option B**: Backend includes portal nested fields in main config

**Chosen**: Option A - Minimal backend changes, frontend adapts to existing API structure.

### Decision 2: How to handle nested config keys?

**Option A**: Add all nested keys to CONFIG_SECTIONS (e.g., `portal.enabled`, `portal.port`)
**Option B**: Use wildcard matching to match any `portal.*` key
**Option C**: Special-case handling in field extraction

**Chosen**: Option A - Explicit is better. Matches existing pattern used by other sections like `discord`, `behavior`.

### Decision 3: Voice fields visibility (API keys are sensitive)

**Option A**: Show voice.key as redacted like other API keys
**Option B**: Allow toggle to show (like discord.bot_token)
**Option C**: Don't include voice.key in editable list at all

**Chosen**: Option C - Voice key is an Azure Speech key that should stay hidden. Only show `voice.region` and `voice.default_voice` as editable.

## Technical Implementation

### Backend Changes (bot/web/routes/config.py)

1. Update `EDITABLE_FIELDS`:
   ```python
   EDITABLE_FIELDS = {
       # ... existing fields ...
       # Voice - add nested fields (excluding sensitive 'key')
       "voice.region",
       "voice.default_voice",
       # Portal - add nested fields
       "portal.enabled",
       "portal.port",
       "portal.logs.retention_days",
       "portal.logs.levels",
       "portal.cors_origins",
   }
   ```

2. Update `read_only_fields`:
   ```python
   read_only_fields = sorted([
       "providers", "models", "fallback_models", "tools",
       "voice",  # Changed from "azure-speech"
       "permissions"
   ])
   ```

### Frontend Changes (web/src/components/ConfigEditor.tsx)

1. Update `CONFIG_SECTIONS`:
   ```typescript
   const CONFIG_SECTIONS = [
     // ... existing sections ...
     { 
       id: 'voice', 
       label: 'Voice', 
       icon: '🎤', 
       fields: ['voice.region', 'voice.default_voice'] 
     },
     { 
       id: 'portal', 
       label: 'Portal', 
       icon: '🌐', 
       fields: [
         'portal.enabled', 
         'portal.port', 
         'portal.cors_origins',
         'portal.logs.retention_days',
         'portal.logs.levels'
       ] 
     },
   ]
   ```

2. Update field extraction in `fetchConfig`:
   - Extract portal from `apiData.portal` instead of `apiData.config.portal`
   - Add special handling for voice fields to match from `apiData.config.voice`

3. Update `renderInput`:
   - Add handling for boolean fields (portal.enabled)
   - Add handling for array fields (portal.logs.levels, portal.cors_origins)

## Risks / Trade-offs

1. **[Risk] Portal logs.levels is an array but UI expects string** → Mitigation: Add array-to-string conversion in renderInput (already exists for lists)

2. **[Risk] Changes to EDITABLE_FIELDS might affect other parts of the system** → Mitigation: Only adding new keys, not modifying existing ones

3. **[Risk] Frontend might not save nested portal/voice changes correctly** → Mitigation: Test save functionality after implementation

## Migration Plan

1. **Deploy Order**:
   - Backend first (adds new editable fields to API)
   - Frontend second (updates ConfigEditor to use them)
   
2. **Rollback**:
   - If frontend causes issues, revert ConfigEditor.tsx changes
   - If backend causes issues, revert EDITABLE_FIELDS changes

3. **Testing**:
   - Verify Portal section shows with all nested fields
   - Verify Voice section shows with region and default_voice
   - Test saving changes and verify they persist to config.yaml