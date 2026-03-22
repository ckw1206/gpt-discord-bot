## Why

The current `config.yaml` has 20+ flat top-level keys (bot_token, client_id, status_message, permissions, providers, models, fallback_models, tools, azure-speech, portal, etc.), making it difficult to navigate and "collapse" sections in YAML-aware editors. Grouping related configs into logical sections improves readability and maintainability.

## What Changes

- **Reorganize config.yaml** into logical sections: `discord`, `behavior`, `llm`, `tools`, `voice`, `portal`
- **Add normalization layer** in config loader that converts old flat keys to new nested structure (with deprecation warnings)
- **Update validator** to support both old and new key paths during migration
- **Update voice config** to accept both `azure-speech` (old) and `voice` (new) keys
- **Update portal API** to support both old and new editable field paths
- **Update config-example.yaml** with new grouped structure
- **Update ConfigEditor.tsx** to display fields grouped by section
- **Update spec** documentation to reflect new structure
- **Keep full backward compatibility**: old keys work during migration period

### NEW: ConfigEditor UI Enhancement (Post-Grouping)

After implementing section grouping in Task 6, user feedback identified areas for improvement:

#### Enhancement 1: New Sidebar Layout
- Replace accordion-style sections with **sidebar + content** layout
- Left sidebar: flat list (Discord, Behavior, LLM, Tools, Voice, Portal)
- Right content: shows fields for selected section
- Full-width input fields to utilize available space

#### Enhancement 2: LLM Section with Tabs
- When "LLM" is selected, show 3 tabs: **Providers | Models | Persona**
- Providers tab: Table editor with Add/Edit/Delete
- Models tab: Table editor with persona dropdown, tool multi-select, toggles
- Persona tab: Simple form for persona, system_prompt, fallback_models

#### Enhancement 3: Table Editors
- **Providers table**: Name, Base URL, API Key, Actions (edit/delete)
- **Models table**: Model Name, Persona (dropdown), Tools (checkbox multi-select), Supports Tools (toggle), Think (toggle), Actions
- Add modal dialogs for Add/Edit operations

#### Enhancement 4: Fix Field Duplication (Completed)
- Root cause: Both legacy and grouped keys present in EDITABLE_FIELDS
- Solution: Remove legacy keys from EDITABLE_FIELDS (legacy auto-converts on load)

#### Enhancement 5: Expand Editable Parameters (Completed)
- Add: `discord.client_id`, `llm.persona`, `llm.system_prompt`, `llm.fallback_models`

#### Enhancement 6: Add JSON/Raw Edit Mode (Deferred)
- Toggle between Form mode and raw JSON editing
- Matches TaskEditor.tsx and PersonaEditor.tsx styling
- Bidirectional sync: Form ↔ JSON preserves all field values
- Validation errors shown inline

#### Enhancement 7: Voice & Portal Tabs (PARTIALLY COMPLETE)

*Status: Tabs added to CONFIG_SECTIONS, backend fields not yet exposed*

**What's done:**
- Voice and Portal tabs exist in sidebar (CONFIG_SECTIONS in ConfigEditor.tsx)
- Tabs are clickable and switch content area

**What's remaining:**
- Backend: Add voice and portal to read_only_fields in config.py
- Backend: Return structured voice config (key, region, default_voice)
- Backend: Return structured portal config (enabled, port, logs.*)
- Frontend: Handle complex nested objects in renderInput for voice/portal sections
- Frontend: Make portal fields editable

**Voice tab fields to display:**
- `voice.key` (sensitive - mask with show/hide toggle)
- `voice.region`
- `voice.default_voice`
- `voice.subscription_key` (sensitive - mask)

**Portal tab fields to display:**
- `portal.enabled` (checkbox)
- `portal.port` (number)
- `portal.require_discord_admin` (checkbox)
- `portal.logs.retention_days` (number)
- `portal.logs.levels` (list/comma-separated)

### Key renames:
- `bot_token`, `client_id`, `status_message`, `permissions` → `discord.*`
- `max_text`, `max_images`, `max_messages`, `use_plain_responses`, `show_embed_color`, `allow_dms` → `behavior.*`
- `providers`, `models`, `fallback_models`, `persona`, `system_prompt` → `llm.*`
- `azure-speech` → `voice`

## Capabilities

### New Capabilities
None - this is a structural refactoring of existing configuration.

### Modified Capabilities
- **config**: Reorganize top-level keys into grouped sections while maintaining backward compatibility

## Impact

- **bot/config/loader.py**: Add `_normalize_config()` function to convert old keys to new structure
- **bot/config/validator.py**: Update validation to check both old and new key paths
- **bot/voice/config.py**: Support both `voice` and `azure-speech` keys
- **bot/web/routes/config.py**: Update EDITABLE_FIELDS to support both paths
- **config.yaml** + **config-example.yaml**: Add new grouped structure (keep old for compat)
- **web/src/components/ConfigEditor.tsx**: Group fields by section in UI
- **openspec/specs/config/spec.md**: Update documentation

### Additional Impact (ConfigEditor Enhancement)

#### Completed Enhancements
- **bot/web/routes/config.py**: 
  - Remove legacy keys from EDITABLE_FIELDS (only grouped keys remain)
  - Add new editable fields (client_id, persona, system_prompt, fallback_models)

#### New Implementation (9.3)
- **web/src/components/ConfigEditor.tsx**: Complete rewrite with:
  - Sidebar layout (flat list: Discord, Behavior, LLM, Tools, Voice, Portal)
  - Full-width input fields
  - LLM section with tabs (Providers | Models | Persona)
  - Providers table: Add/Edit/Delete with modal dialogs
  - Models table: Add/Edit/Delete with modal dialogs
    - Persona dropdown
    - Tools multi-select checkbox list
    - Supports Tools toggle
    - Think toggle

#### Deferred (9.4 - JSON Mode)
- Add edit mode toggle (Form/JSON)
- Section selector dropdown for JSON mode
- Implement bidirectional Form ↔ JSON sync
- Style JSON textarea to match TaskEditor.tsx

---

### Post-Enhancement Bug Fixes & UI Improvements

*Issues identified from user testing on 2026-03-20*

#### Bug 1: client_id Display Mismatch
- **Issue**: `discord.client_id` shows wrong value (1470091668081479700 vs actual 1470091668081479690)
- **Root Cause**: `discord.client_id` not in EDITABLE_FIELDS, so UI shows default/stale value instead of current config
- **Fix**: Add `discord.client_id` and `discord.bot_token` to EDITABLE_FIELDS

#### Bug 2: Save/Apply Button Errors
- **Issue**: "Failed to save config" / "Failed to apply config" without details
- **Root Cause**: Error messages don't include backend response details
- **Fix**: Update toast error to include `err.response?.data?.detail`

#### Bug 3: Delete In-Memory Provider Fails
- **Issue**: Cannot delete provider that was just added
- **Root Cause**: State handling issue when provider list becomes empty or has null values

#### Bug 4: Model Name Not Editable
- **Issue**: Model name field is disabled in Models table
- **Fix**: Enable editing of model name field

#### UI Issue 1: Input Field Spacing
- **Issue**: Input fields touch right edge of container
- **Fix**: Add 16px right margin to input fields

#### UI Issue 2: Number Input Spinners
- **Issue**: Browser-default up/down arrows show on number inputs
- **Fix**: Hide spinners with CSS

#### UI Issue 3: Checkbox Alignment
- **Issue**: Checkboxes (allow_dms, show_embed_color) are centered vertically
- **Fix**: Align checkboxes to right side of label

#### UI Issue 4: API Key Visibility
- **Issue**: API keys in provider table are always visible
- **Fix**: Add view/hide toggle button for API key fields

#### UI Issue 5: Unnecessary Tabs
- **Issue**: "Persona" sub-tab in LLM section is redundant (persona is per-model)
- **Issue**: "Tools" tab is too complex for simple editing
- **Fix**: Remove Persona sub-tab, remove Tools tab

#### UI Issue 6: Missing Tabs
- **Issue**: Voice and Portal sections not accessible via sidebar
- **Fix**: Add Voice and Portal tabs to sidebar

---

### Key renames: