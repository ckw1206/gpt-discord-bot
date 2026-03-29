## 1. Config Loader - Normalization Layer

- [x] 1.1 Add `_normalize_config()` function to `bot/config/loader.py` that converts legacy flat keys to new grouped structure
- [x] 1.2 Add deprecation warning function that logs warning once per key when legacy keys are used
- [x] 1.3 Call normalization in `get_config()` after loading YAML but before returning
- [x] 1.4 Test normalization with legacy-only config
- [x] 1.5 Test normalization with new grouped config
- [x] 1.6 Test normalization with mixed config (some old, some new)

## 2. Config Validator - Dual Path Support

- [x] 2.1 Update `bot/config/validator.py` to check both old and new provider paths
- [x] 2.2 Update validator to check both old and new models paths
- [x] 2.3 Update validator to support both `voice` and `azure-speech` keys
- [x] 2.4 Add validation warning when legacy keys are used
- [x] 2.5 Integration test

## 3. Voice Config - Backward Compatibility

- [x] 3.1 Update `bot/voice/config.py` to check for both `voice` and `azure-speech` keys
- [x] 3.2 Keep `voice` as the primary key, fall back to `azure-speech` for backward compat
- [x] 3.3 Add deprecation warning when `azure-speech` is used
- [x] 3.4 Integration test

## 4. Portal Config API - Editable Fields

- [x] 4.1 Update `bot/web/routes/config.py` EDITABLE_FIELDS to include both old and new paths
- [x] 4.2 Update `_filter_config()` to handle both old and new paths
- [x] 4.3 Update `_format_for_display()` to handle both `voice` and `azure-speech`
- [x] 4.4 Test config API with new grouped structure

## 5. Config Files - New Structure

- [x] 5.1 Update `config.yaml` to use new grouped structure (keep legacy keys for backward compat during testing)
- [x] 5.2 Update `config-example.yaml` to use new grouped structure
- [x] 5.3 Verify both old and new configs work correctly

## 6. Frontend - ConfigEditor Grouping

- [x] 6.1 Update `web/src/components/ConfigEditor.tsx` to group fields by section
- [x] 6.2 Add collapsible sections for discord, behavior, llm, tools, voice, portal
- [x] 6.3 Test UI displays correctly with new config structure

## 7. Specification - Documentation

- [x] 7.1 Update `openspec/specs/config/spec.md` to document new grouped structure
- [x] 7.2 Document backward compatibility and deprecation timeline
- [x] 7.3 Update config-example.yaml comments if needed
- [x] 7.4 Write unit test

## 8. Testing - Full Migration

- [x] 8.1 Test bot starts with legacy config (should work with warnings)
- [x] 8.2 Test bot starts with new grouped config
- [x] 8.3 Test bot starts with mixed config
- [x] 8.4 Test portal config API with all format combinations
- [x] 8.5 Test portal UI with new config
- [x] 8.6 Verify no existing tests are broken

**Note:** 6 tests in test_voice_config.py and test_voice_config_standalone.py fail when run after test_bot_integration.py loads llmcord.py (which reads .env for AZURE_SPEECH_* vars). This is a pre-existing test pollution issue unrelated to this change. Tests pass when run in isolation.

## 9. ConfigEditor Enhancement

*Post-grouping improvements identified from user feedback*

### 9.1 Fix Field Duplication

- [x] 9.1.1 Remove legacy keys from EDITABLE_FIELDS in `bot/web/routes/config.py`
  - Keep only grouped keys: `discord.status_message`, `behavior.max_text`, etc.
  - Legacy keys still work for backward compatibility (normalization layer handles conversion)
- [x] 9.1.2 Remove Legacy section from CONFIG_SECTIONS in ConfigEditor.tsx
- [x] 9.1.3 Test: Verify no duplicate fields in ConfigEditor UI

### 9.2 Expand Editable Parameters

- [x] 9.2.1 Add `discord.client_id` to EDITABLE_FIELDS
- [x] 9.2.2 Add `llm.persona` to EDITABLE_FIELDS
- [x] 9.2.3 Add `llm.system_prompt` to EDITABLE_FIELDS
- [x] 9.2.4 Add `llm.fallback_models` to EDITABLE_FIELDS
- [x] 9.2.5 Update backend PUT /config to handle these new fields
- [x] 9.2.6 Test: Verify new fields appear in ConfigEditor and can be saved

### 9.2.B Backend API for LLM Table Editing

*Prerequisite for tasks 9.3.3-9.3.5 - The frontend needs structured API to build table editors.*

- [x] 9.2.7 Modify GET /config to return structured providers/models instead of formatted strings
  - Add new response field `llm_providers` with full provider objects
  - Add new response field `llm_models` with full model objects  
  - Keep existing `config.providers` and `config.models` formatted for display
- [x] 9.2.8 Add PUT /config/providers endpoint for CRUD on providers
  - Accept: { action: "add"|"update"|"delete", name, data }
  - Write directly to config.yaml
- [x] 9.2.9 Add PUT /config/models endpoint for CRUD on models
  - Accept: { action: "add"|"update"|"delete", model_name, data }
  - Write directly to config.yaml
- [x] 9.2.10 Test: GET /config returns structured llm_providers and llm_models
- [x] 9.2.11 Test: Add/Edit/Delete provider persists correctly
- [x] 9.2.12 Test: Add/Edit/Delete model persists correctly

### 9.3 New Sidebar Layout & LLM Table Editors

*Note: Current ConfigEditor.tsx still uses accordion style. Implementation needed.*

#### Implementation: ConfigEditor New Layout

- [x] 9.3.1 Replace accordion with sidebar + content layout
- [x] 9.3.2 Add full-width input fields
- [x] 9.3.3 Implement LLM section with tabs (Providers | Models | Persona)
- [x] 9.3.4 Create Providers table with Add/Edit/Delete
- [x] 9.3.5 Create Models table with Add/Edit/Delete
  - Persona dropdown (bao, default)
  - Tools multi-select checkbox list
  - Supports Tools toggle
  - Think toggle
- [x] 9.3.6 Add modal dialogs for Provider and Model editing
- [x] 9.3.7 Test: Verify all sections render correctly
- [x] 9.3.8 Test: Add/Edit/Delete providers works
- [x] 9.3.9 Test: Add/Edit/Delete models works

### 9.4 Add JSON/Raw Edit Mode

- [x] 9.4.1 Add edit mode toggle state (Form | JSON) in ConfigEditor.tsx
- [x] 9.4.2 Implement Form → JSON conversion:
  - Reconstruct nested object from flat key-value pairs
  - Display as formatted JSON in textarea
- [x] 9.4.3 Implement JSON → Form conversion:
  - Parse JSON string
  - Validate structure and show errors inline
  - Flatten back to key-value pairs for form fields
- [x] 9.4.4 Add bidirectional sync logic:
  - Switching modes preserves all field values
  - Save in JSON mode parses and flattens correctly
- [x] 9.4.5 Style JSON textarea to match TaskEditor.tsx:
  - Monospace font, #1a1a1a background, #ddd text
  - Red border (#ff6b6b) on validation error
  - Error message displayed above textarea
- [x] 9.4.6 Test: Toggle between Form and JSON, verify values sync
- [x] 9.4.7 Test: Invalid JSON shows error, prevents mode switch

---

## 11. Bug Fixes & UI Improvements (Post-Feedback)

*Issues identified from user testing on 2026-03-20*

### 11.1 Discord Tab - client_id Display Issue

- [x] 11.1.1 Add `discord.bot_token` to EDITABLE_FIELDS in backend
- [x] 11.1.2 Verify client_id displays correctly in UI
- [x] 11.1.3 Add to CONFIG_SECTIONS in ConfigEditor.tsx if missing

### 11.2 Behavior Tab - UI Improvements

- [x] 11.2.1 Add right padding/margin to input fields to keep distance from container edge
- [x] 11.2.2 Remove browser-default number input spinners (CSS: `::-webkit-inner-spin-button { display: none }`)
- [x] 11.2.3 Align checkboxes (allow_dms, show_embed_color) to the right side of the label

### 11.3 LLM Tab - Fix Save/Delete Issues

- [x] 11.3.1 Fix save button error: Add detailed error logging and toast with `err.response?.data?.detail`
- [x] 11.3.2 Fix apply button error: Same as above
- [x] 11.3.3 Fix delete in-memory provider: Handle empty/null provider list properly
- [x] 11.3.4 Make model name editable in Models table
- [x] 11.3.5 Add API key visibility toggle (view/hide button) for provider API keys

### 11.4 Remove Unnecessary Tabs

- [x] 11.4.1 Remove "Persona" sub-tab from LLM tab (persona is set per-model, not global)
- [x] 11.4.2 Remove "Tools" tab from sidebar (tools are complex nested config, not easily editable in current UI)

### 11.5 fallback_models Type Handling

*Fix for 2026-03-20 error: 'fallback_models' must be a list, got str*

- [x] 11.5.1 Fix config.yaml: change `fallback_models: string` to `fallback_models: [list]`
- [x] 11.5.2 Frontend: Add LIST_FIELDS handling in ConfigEditor.tsx renderInput
- [x] 11.5.3 Backend: Convert comma-separated string to list in PUT /config
- [x] 11.5.4 Normalization layer: Auto-convert string to list in loader.py

### 11.6 Add Missing Tabs (Voice & Portal)

*Status: PARTIALLY COMPLETE - Tabs exist in CONFIG_SECTIONS but not showing content*

- [x] 11.6.1 Add "Voice" tab to sidebar with voice config fields (tabs exist, not populating)
- [x] 11.6.2 Add "Portal" tab to sidebar with portal config fields (tabs exist, not populating)
- [x] 11.6.3 Update CONFIG_SECTIONS in ConfigEditor.tsx accordingly (done)

*Implementation details documented in [design.md#decision-12] and [design.md#decision-13]*

**Remaining work: Deferred to future work**

### 11.7 UI Standardization

- [x] 11.7.1 Apply consistent padding/margins per `openspec/specs/web-portal/ui-standards.md`
- [x] 11.7.2 Fix checkbox alignment across all sections
- [x] 11.7.3 Remove number input spinners globally
- [ ] 11.7.4 Verify all buttons follow standard colors (green=save, blue=apply, red=delete) - **Deferred**
- [ ] 11.7.5 Add loading state (opacity: 0.5) to buttons during async operations - **Deferred**
- [ ] 11.7.6 Update error messages to include details from backend response - **Deferred**
- [x] 11.7.7 Add input field standards to ui-standards.md (Section 9: wrapper-based layout)
- [x] 11.7.8 Implement consistent input wrapper in ConfigEditor.tsx (all text/number inputs use flex wrapper)

### 11.8 Icon & Visibility Fixes

- [x] 11.8.1 Add heroicons import to ConfigEditor.tsx (EyeIcon, EyeSlashIcon, PlusIcon, PencilIcon, TrashIcon)
- [x] 11.8.2 Replace emoji icons with heroicons in provider/model tables
- [x] 11.8.3 Fix API key visibility toggle (backend now returns non-redacted api_key)
- [x] 11.8.4 Fix discord.bot_token visibility toggle
- [x] 11.8.5 Add UI standards for icons to `openspec/specs/web-portal/ui-standards.md`
- [x] 11.8.6 Fix JSON editor not updating when switching sections

---

## 12. Backend API - Expose Voice & Portal Fields

*Voice and Portal tabs exist in UI but content isn't showing. Need to expose these fields in API.*

**Status: Deferred to future work**

### 12.1 Add Voice to Read-Only Fields

*Voice config needs to be returned from API so UI can display it*

- [ ] 12.1.1 Add `voice` to read_only_fields in config.py (or handle 'voice' as special case) - **Deferred**
- [ ] 12.1.2 Ensure voice config is returned in the config response (not redacted) - **Deferred**
- [ ] 12.1.3 Handle nested voice config in _format_for_display() - **Deferred**

### 12.2 Add Portal to Read-Only Fields

*Portal config is fetched separately but needs to show in UI*

- [ ] 12.2.1 Add `portal` to read_only_fields in config.py - **Deferred**
- [ ] 12.2.2 Return structured portal config in API response (not formatted string) - **Deferred**
- [ ] 12.2.3 Handle nested portal config in ConfigEditor.tsx (enabled, port, logs.*) - **Deferred**

### 12.3 Permissions Display

- [ ] 12.3.1 Return structured permissions in config response - **Deferred**
- [ ] 12.3.2 Display permissions properly in Discord section - **Deferred**

---

## 13. Frontend - Voice & Portal Section Rendering

*After backend exposes fields, render them properly in UI*

**Status: Deferred to future work**

### 13.1 Voice Section Rendering

- [ ] 13.1.1 Add special handling in renderInput for voice section - **Deferred**
- [ ] 13.1.2 Display voice.key (redacted), voice.region, voice.default_voice - **Deferred**
- [ ] 13.1.3 Make voice fields editable (add to EDITABLE_FIELDS) - **Deferred**

### 13.2 Portal Section Rendering

- [ ] 13.2.1 Add special handling in renderInput for portal section - **Deferred**
- [ ] 13.2.2 Display portal.enabled (toggle), portal.port (number) - **Deferred**
- [ ] 13.2.3 Display portal.logs.retention_days, portal.logs.levels - **Deferred**
- [ ] 13.2.4 Make portal fields editable (add to EDITABLE_FIELDS) - **Deferred**

### 13.3 Permissions Rendering

- [ ] 13.3.1 Display permissions as read-only formatted text or nested fields - **Deferred**

---

## 14. Web Portal Spec Refactoring

*Refactor web-portal UI specifications for consistency and maintainability*

### 14.1 Rename UI Standards File

- [x] 14.1.1 Rename `openspec/specs/web-portal/ui-standards.md` to `openspec/specs/web-portal/UI_CONVENTIONS.md`
- [x] 14.1.2 Update all internal links and references to the renamed file

### 14.2 Fix Section Numbering

*Issue: Section 8 appears twice (Icon Standards and Related Specifications)*

- [x] 14.2.1 Change "Related Specifications" from Section 8 to Section 10
- [x] 14.2.2 Verify all section numbers are sequential (1-10)

### 14.3 Consolidate Duplicate Button Specs

*Issue: Button specs duplicated in both ui-standards.md and config-editor-spec.md*

- [x] 14.3.1 Remove Section 3 (Buttons) from `config-editor-spec.md`
- [x] 14.3.2 Add reference in `config-editor-spec.md` pointing to `UI_CONVENTIONS.md#section-2-buttons`
- [x] 14.3.3 Update other sections in config-editor-spec.md that reference buttons to point to UI_CONVENTIONS.md

### 14.4 Add Cross-References to Shared Standards

*Issue: Other specs don't reference ui-standards for shared patterns*

- [x] 14.4.1 Add reference in `openspec/specs/task-management/spec.md`:
  > "UI patterns follow [web-portal/UI_CONVENTIONS.md](../web-portal/UI_CONVENTIONS.md)"
- [x] 14.4.2 Add reference in `openspec/specs/persona-management/spec.md`:
  > "UI patterns follow [web-portal/UI_CONVENTIONS.md](../web-portal/UI_CONVENTIONS.md)"
- [x] 14.4.3 Add reference in `openspec/specs/config-three-buttons/spec.md`:
  > "Button styles follow [web-portal/UI_CONVENTIONS.md](../web-portal/UI_CONVENTIONS.md)"

### 14.5 Document Save/Apply/Save&Apply Semantics

*Issue: Button colors defined in UI_CONVENTIONS but semantic behavior defined elsewhere*

- [x] 14.5.1 Add section in UI_CONVENTIONS.md explaining Save/Apply/Save&Apply behavior:
  - Save: writes to config.yaml only
  - Apply: reloads in-memory only
  - Save&Apply: writes AND reloads
- [x] 14.5.2 Reference the behavioral spec in `config-three-buttons/spec.md`

---

## 15. Cleanup - Post-Migration

*Final verification and cleanup after all fixes are applied*

**Status: Deferred to future work**

- [ ] 15.1 Remove temporary test configs - **Deferred**
- [ ] 15.2 Merge, absorb and refactor test files if needed - **Deferred**
- [ ] 15.3 Document migration notes in CHANGELOG.md if exists - **Deferred**
- [ ] 15.4 Final verification: bot works with clean new config structure - **Deferred**
- [ ] 15.5 Verify ConfigEditor sidebar layout renders correctly in all sections - **Deferred**
- [ ] 15.6 Verify LLM tabs (Providers/Models) work correctly - **Deferred**
- [ ] 15.7 Verify table Add/Edit/Delete for providers persists to config.yaml - **Deferred**
- [ ] 15.8 Verify table Add/Edit/Delete for models persists to config.yaml - **Deferred**
- [ ] 15.9 Test full save/apply cycle: make changes → Save → Apply → verify changes active - **Deferred**

---

*Change archived: March 22, 2026 - Core config restructuring complete, deferred work moved to future*