## Context

**Current State:**
- `config.yaml` has 20+ flat top-level keys
- All config access goes through `bot/config/loader.py`'s `get_config()` function
- Config is accessed directly in `llmcord.py` (~50+ accesses), validator, voice config, web routes, and tools

**Constraints:**
- Must maintain full backward compatibility during migration
- Portal config editor must continue to work
- No breaking changes to existing deployments
- Existing tests must continue to pass

**Stakeholders:**
- Users editing config.yaml manually
- Web portal users using ConfigEditor
- Future maintainers

## Goals / Non-Goals

**Goals:**
- Group related config keys into logical sections (discord, behavior, llm, tools, voice, portal)
- Add normalization layer that converts old flat keys to new nested structure
- Provide deprecation warnings when old keys are used
- Update portal config API to support both old and new paths
- Update UI to display fields grouped by section

**Non-Goals:**
- Removing old key support (deferred to future release)
- Changing config validation rules (only structural changes)
- Adding new configuration options

## Decisions

### Decision 1: Normalize to new structure on load

**Choice:** Add `_normalize_config()` in `loader.py` that converts old keys to new nested structure.

**Rationale:** 
- Single point of change: all code sees the normalized structure
- Simpler code in `llmcord.py` and elsewhere - no need to update ~50+ direct accesses
- Can add deprecation warnings in one place

**Alternative Considered:** Update all direct `config["key"]` accesses in llmcord.py to use new paths.
- **Rejected:** Too many changes, high risk of missing one, harder to maintain backward compat during migration.

### Decision 2: Support both old and new keys in validator

**Choice:** Validator checks for both old and new paths (e.g., `cfg.get("llm.providers") or cfg.get("providers")`).

**Rationale:**
- Users can migrate gradually - use new keys, old keys, or mix
- No need to force immediate rewrite of config files

### Decision 3: Rename `azure-speech` to `voice`

**Choice:** Add new `voice` section, keep `azure-speech` as deprecated alias.

**Rationale:**
- `azure-speech` is Azure-specific; `voice` is more generic (future-proof)
- Allow both during migration

### Decision 4: Portal API supports dot-notation for both paths

**Choice:** `EDITABLE_FIELDS` includes both `"status_message"` and `"discord.status_message"`.

**Rationale:**
- Frontend sends dot-notation paths; backend needs to handle both
- User can edit using either old or new path during migration

## Risks / Trade-offs

### Risk 1: Normalization logic could miss some keys
**Mitigation:** Comprehensive testing with both old-only, new-only, and mixed configs.

### Risk 2: Deprecation warnings could be noisy
**Mitigation:** Only warn once per key, not on every config load.

### Risk 3: ConfigEditor UI changes might confuse users
**Mitigation:** Group fields visually by section; show both old and new paths in field names.

---

## New Section: ConfigEditor UI Enhancement Design

### Context

After implementing section grouping in Task 6, user testing revealed:
1. **Layout issue**: Accordion-style sections don't utilize horizontal space efficiently
2. **LLM complexity**: Providers and models have nested structure, need dedicated editors
3. **Duplication issue**: `status_message` appears in both "Discord" and "Legacy" tabs
4. **Limited editability**: Only a subset of config fields can be edited via UI
5. **Power user need**: Advanced users want to edit raw JSON for complex nested configs

---

### Decision 8: Sidebar + Content Layout

**Choice:** Replace accordion with sidebar navigation + content panel.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ⚙️ Config Editor                                      [Save] [Apply]  │
├──────────────────┬──────────────────────────────────────────────────────┤
│                  │                                                       │
│  💬 Discord      │   💬 Discord                                          │
│  ⚙️ Behavior     │   ─────────────────────────────────────────────────  │
│  🤖 LLM          │                                                       │
│  🔧 Tools        │   status_message    [________________________]      │
│  🎤 Voice        │   client_id         [________________________]      │
│  🌐 Portal       │   bot_token         [________________________]      │
│                  │                                                       │
└──────────────────┴──────────────────────────────────────────────────────┘
```

**Rationale:**
- Better utilization of horizontal space
- Clearer navigation with active state indicator
- Full-width input fields for better UX

---

### Decision 9: LLM Section with Tabs

**Choice:** When LLM is selected, show tabbed interface: Providers | Models | Persona

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🤖 LLM                                               [Save] [Apply]   │
├──────────────────┬──────────────────────────────────────────────────────┤
│  ...             │   ┌──────────────┬──────────────┬──────────────┐    │
│                  │   │ Providers    │ Models       │ Persona      │    │
│                  │   ├──────────────┼──────────────┼──────────────┤    │
│                  │   │ [TABLE]      │ [TABLE]      │ [FORM]       │    │
│                  │   │              │              │              │    │
│                  │   └──────────────┴──────────────┴──────────────┘    │
└──────────────────┴──────────────────────────────────────────────────────┘
```

**Providers Tab Table:**
| Name | Base URL | API Key | Actions |
|------|----------|---------|---------|
| google | https://... | •••••••• | ✏️ 🗑️ |
| ollama | http://... | (empty) | ✏️ 🗑️ |

**Models Tab Table:**
| Model Name | Persona | Tools | Supports Tools | Think | Actions |
|------------|---------|-------|----------------|-------|---------|
| ollama/... | bao ▼ | ☑ web_search, ... | ✓ | ✓ | ✏️ 🗑️ |

**Rationale:**
- Providers and models are complex nested structures
- Table view makes it easy to scan and compare
- Modal dialogs for add/edit keep UI clean

---

### Decision 10: Table Editor Implementation

**Choice:** Inline table with modal dialogs for CRUD operations.

**Provider Modal:**
- Name (text input)
- Base URL (text input)
- API Key (text input)

**Model Modal:**
- Model Name (text input)
- Persona (dropdown: bao, default)
- Supports Tools (checkbox)
- Tools (checkbox list: web_search, get_market_prices, get_weather, google_tools, visuals_core)
- Think (checkbox)

**Rationale:**
- Matches data structure in config.yaml
- Checkbox multi-select for tools is intuitive
- Dropdown prevents typos in persona names

---

### Decision 11: Backend API for Structured LLM Data

**Problem Identified:**
- Current `/config` endpoint returns `providers` and `models` as formatted strings (e.g., "google, ollama, openrouter")
- Frontend cannot build table editors without the full nested data structure
- Table CRUD operations need dedicated API endpoints

**Solution:** Extend config API with structured data endpoints.

**Choice:** Add new API endpoints alongside existing `/config`:
1. GET /config returns new fields: `llm_providers` and `llm_models` (full objects)
2. PUT /config/providers - CRUD for providers
3. PUT /config/models - CRUD for models

**Alternative Considered:** Parse formatted strings in frontend
- **Rejected:** Fragile - would break if display format changes
- Frontend needs reliable, structured data for table operations

**Implementation:**
```python
# GET /config response - NEW fields
{
  "config": { ... },
  "llm_providers": {
    "google": { "base_url": "...", "api_key": "..." },
    "ollama": { "base_url": "...", "api_key": "" }
  },
  "llm_models": {
    "ollama/qwen3:14b": { "persona": "bao", "supports_tools": true, "tools": [...], "think": true }
  },
  "editable_fields": [...],
  "read_only_fields": [...]
}

# PUT /config/providers
{ "action": "add"|"update"|"delete", "name": "google", "data": { ... } }

# PUT /config/models  
{ "action": "add"|"update"|"delete", "model_name": "ollama/qwen3:14b", "data": { ... } }
```

**Rationale:**
- Keeps existing API backward compatible (adds new fields, doesn't change existing)
- Follows REST-like patterns already used in the codebase
- Direct YAML write ensures persistence to config.yaml

---

### Decision 12: Voice Section Implementation

Voice fields needed in backend `read_only_fields`:
- voice.key
- voice.region  
- voice.default_voice
- voice.subscription_key (sensitive - redact)

The CONFIG_SECTIONS already has the voice section configured:
```tsx
{ id: 'voice', label: 'Voice', icon: '🎤', fields: ['voice', 'azure-speech'] },
```

**Implementation Tasks:**
- 11.6.4 Add voice fields to backend `read_only_fields`
- 11.6.5 Return structured voice config in API response
- 11.6.7 Handle complex object display in ConfigEditor for voice section

---

### Decision 13: Portal Section Implementation

Portal fields needed in backend:
- Currently handled separately via `_get_portal_config_safe()` 
- Need to expose structured portal fields in the config response

The CONFIG_SECTIONS already has the portal section configured:
```tsx
{ id: 'portal', label: 'Portal', icon: '🌐', fields: ['portal'] },
```

**Implementation Tasks:**
- 11.6.6 Return structured portal config in API response (or expose portal.* fields)
- 11.6.7 Handle complex object display in ConfigEditor for portal section

---

### Decision 5: Remove Legacy Keys from EDITABLE_FIELDS

**Choice:** Only include grouped (new) keys in EDITABLE_FIELDS; remove legacy equivalents.

**Rationale:**
- Legacy keys are for **backward-compatible loading only**, not editing
- Users should edit using the canonical new paths
- Backend normalization layer handles legacy → grouped conversion automatically

**Implementation:**
```python
# bot/web/routes/config.py
EDITABLE_FIELDS = {
    # Only new grouped keys
    "discord.status_message", "discord.client_id",
    "behavior.max_text", "behavior.max_images", ...
    "llm.persona", "llm.system_prompt", "llm.fallback_models",
    "portal",
}
# Legacy keys removed - they auto-convert on load
```

---

## Post-Enhancement Bug Fixes & UI Improvements Design

### Issue 1: client_id Display Mismatch

**Problem:** UI shows wrong value for `discord.client_id` (1470091668081479700 vs actual 1470091668081479690 in config.yaml).

**Root Cause Analysis:**
- `discord.client_id` was added to EDITABLE_FIELDS in task 9.2.1
- However, the UI was showing a default/stale value because the config wasn't being loaded correctly
- The issue is that client_id is in read-only fields, not editable fields in the current config response

**Solution:** Ensure client_id is properly loaded and displayed in the UI.

---

### Issue 2: Save/Apply Button Errors Without Details

**Problem:** Error toast shows "Failed to save config" without details from the backend.

**Root Cause Analysis:**
- Current error handling: `toast.error('Failed to save config')`
- Missing: `err.response?.data?.detail` extraction

**Solution:**
```typescript
// Current (broken)
toast.error('Failed to save config')

// Fixed
toast.error(err.response?.data?.detail || 'Failed to save config')
```

---

### Issue 3: UI Spacing and Alignment Issues

**Problem:** Input fields touch right edge, number spinners visible, checkboxes misaligned.

**Solution:** Apply CSS rules per ui-standards.md:

```css
/* Input fields */
input[type="text"],
input[type="number"],
select {
  margin-right: 16px;  /* Keep distance from container edge */
}

/* Number input spinners */
input[type="number"]::-webkit-inner-spin-button {
  -webkit-appearance: none;
}

/* Checkbox alignment */
label {
  display: flex;
  align-items: center;
}
```

---

### Issue 4: Unnecessary Tabs (Persona sub-tab, Tools tab)

**Problem:** 
- Persona sub-tab in LLM section is redundant (persona is set per-model, not global)
- Tools tab is too complex for simple editing

**Solution:** Remove from CONFIG_SECTIONS:
- Remove "Persona" from LLM_TABS array
- Remove "Tools" from CONFIG_SECTIONS

---

### Issue 5: Missing Voice and Portal Tabs

**Problem:** Voice and Portal sections exist in config but aren't accessible via sidebar.

**Solution:** Add to CONFIG_SECTIONS:
```typescript
const CONFIG_SECTIONS = [
  { id: 'discord', label: 'Discord', icon: '💬', fields: [...] },
  { id: 'behavior', label: 'Behavior', icon: '⚙️', fields: [...] },
  { id: 'llm', label: 'LLM', icon: '🤖', fields: [...] },
  { id: 'voice', label: 'Voice', icon: '🎤', fields: ['voice.key', 'voice.region', 'voice.default_voice'] },
  { id: 'portal', label: 'Portal', icon: '🌐', fields: ['portal.enabled', 'portal.port', 'portal.logs'] },
]
```

---

### Issue 6: API Key Visibility Toggle

**Problem:** API keys are always visible in provider table (security concern).

**Solution:** Add show/hide toggle:
```typescript
const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({})

// In provider row:
<input 
  type={showApiKey[provider.name] ? "text" : "password"}
  value={provider.api_key}
/>
<button onClick={() => setShowApiKey(prev => ({...prev, [provider.name]: !prev[provider.name]}))}>
  {showApiKey[provider.name] ? "👁️" : "👁️‍🗨️"}
</button>
```

---

### Issue 7: Model Name Not Editable

**Problem:** Model name field is disabled in the Models table.

**Solution:** Enable the model name input field in the table:
```typescript
// In table row:
<input 
  type="text" 
  value={model.name}
  onChange={(e) => handleModelNameChange(modelIndex, e.target.value)}
  disabled={false}  // Enable editing
/>
```

---

## 12. UI Design Specification (Referenced from tasks.md)

*Supplemental to openspec/specs/web-portal/ui-standards.md*
*See openspec/specs/web-portal/config-editor-spec.md for complete specification*

### 12.1 ConfigEditor Layout Standards

**Sidebar Layout:**
- Width: 200px fixed
- Background: #1a1a1a
- Item padding: 12px 16px
- Active item: #2d2d2d background, left border accent

**Content Area:**
- Padding: 24px
- Max-width: 800px (for form fields)
- Gap between fields: 16px

**Input Fields:**
- Height: 40px
- Padding: 8px 12px
- Border: 1px solid #404040
- Border-radius: 6px
- Focus border: #3b82f6
- Right margin: 16px from container edge
- Remove number spinners: `input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; }`

**Checkboxes:**
- Align: left side of label
- Label text: right of checkbox
- Gap: 8px between checkbox and label text

**Buttons:**
- Height: 36px
- Padding: 0 16px
- Border-radius: 6px
- Font-weight: 500

**Modals:**
- Backdrop: rgba(0, 0, 0, 0.7)
- Width: max 500px
- Border-radius: 8px
- Close button: top-right corner

### 12.2 LLM Table Standards

**Providers Table:**
- Columns: Name, Base URL, API Key (masked), Actions
- API Key field: show "••••••••" by default, eye icon to reveal
- Actions: Edit, Delete buttons per row
- Add button: above table

**Models Table:**
- Columns: Name (editable), Persona (dropdown), Supports Tools (checkbox), Think (checkbox), Tools (multi-select), Actions
- Persona dropdown: options from persona config
- Tools: checkbox list from AVAILABLE_TOOLS
- Actions: Edit, Delete buttons per row
- Add button: above table

### 12.3 Missing Sections to Implement

**Voice Tab Fields:**
- voice.key (sensitive - show as redacted or masked)
- voice.region
- voice.default_voice

**Portal Tab Fields:**
- portal.enabled (checkbox)
- portal.port (number)
- portal.cors_origins (list)
- portal.logs.retention_days (number)
- portal.logs.levels (list)

---

### Decision 6: Expand Editable Fields (Completed)

**Choice:** Add more simple-field parameters to EDITABLE_FIELDS.

**Rationale:**
- `client_id`, `persona`, `system_prompt` are simple strings - easy to edit
- Complex nested objects (permissions, providers, models) require JSON mode

**Added fields:**
- `discord.client_id` - string
- `llm.persona` - string  
- `llm.system_prompt` - string (multi-line)
- `llm.fallback_models` - array

---

### Decision 11: Dual-Mode JSON Editor

**Choice:** Add toggle between Form mode (key-value inputs) and JSON mode (raw textarea).

**Rationale:**
- Matches existing TaskEditor.tsx and PersonaEditor.tsx patterns
- Power users can edit complex nested configs directly
- Maintains accessibility for basic users who prefer form inputs

**UI Specification:**

| Element | Specification |
|---------|---------------|
| Toggle | Segmented control: [Form] [JSON] |
| Section selector | Dropdown to choose which section to edit |
| JSON textarea | Monospace, #1a1a1a bg, #ddd text, 8px radius |
| Error state | #ff6b6b border, error message above textarea |
| Sync behavior | Form → JSON reconstructs nested; JSON → Form flattens |

**Sync Flow:**
```
Form Mode ──(toggle)──▶ JSON Mode
    ▲                      │
    │ 1. Get all fields    │ 1. Parse JSON string
    │ 2. Reconstruct       │ 2. Validate structure
    │    nested object     │ 3. If error: show inline
    │ 3. Stringify (2 spaces)│    stay in JSON mode
    │ 4. Display in        │ 4. If valid: flatten to
    │    textarea          │    key-value pairs
    │                      ▼
    └─────────────────── JSON Mode
              (on valid save)
```

### Risk 4: JSON parse errors could lose data
**Mitigation:** 
- Show validation errors inline before allowing mode switch
- Always confirm before discarding unsaved JSON changes
- Save preserves parsed values back to form state

## Migration Plan

### Phase 1: Add normalization (non-breaking)
1. Add `_normalize_config()` to loader.py
2. Test with existing config (should work unchanged)
3. Add deprecation warnings for old keys

### Phase 2: Add new structure to config files
1. Update config.yaml with new grouped sections (keep old keys)
2. Update config-example.yaml with new structure
3. Validator now accepts both formats

### Phase 3: Update portal
1. Update API to accept both old and new editable field paths
2. Update ConfigEditor to group fields visually

### Phase 4: Document and deprecate
1. Update spec.md with new structure
2. Log warnings when old keys used

## Open Questions

1. **Should we also support environment variable prefixes?** (e.g., `LLM_PROVIDERS_...` vs `PROVIDERS_...`)
   - **Decision:** No, keep current `${VAR}` interpolation as-is.

2. **How long to keep backward compatibility?**
   - **Decision:** At least 2 releases. Remove old keys in next major version.