## 1. Button Styling (Section 2, 16, 19 of UI_CONVENTIONS)

- [x] 1.1 Update Save button: add backgroundColor: '#22c55e', color: '#fff'
- [x] 1.2 Update Apply button: add backgroundColor: '#3b82f6', color: '#fff'
- [x] 1.3 Update Save&Apply button: add backgroundColor: '#22c55e', color: '#fff'
- [x] 1.4 Add button dimensions: height 36px, min-width 80px, gap 12px between buttons
- [x] 1.5 Add loading state: opacity 0.5 when saving=true, change text to 'Saving...'
- [x] 1.6 Update modal Cancel button: add backgroundColor: '#4b5563', color: '#fff'
- [x] 1.7 Update modal Save button in modals: add backgroundColor: '#22c55e', color: '#fff'

## 2. Sidebar (Section 10 of UI_CONVENTIONS)

- [x] 2.1 Change sidebar width from 160px to 200px
- [x] 2.2 Update sidebar item padding to 12px (from 0.75rem)
- [x] 2.3 Verify sidebar border-radius is 8px

## 3. Input Fields (Section 9 of UI_CONVENTIONS)

- [x] 3.1 Add input field height: 40px
- [x] 3.2 Add input backgroundColor: '#1a1a1a'
- [x] 3.3 Add input border: '1px solid #404040'
- [x] 3.4 Add input border-radius: 6px
- [x] 3.5 Add input text color: '#fff'
- [x] 3.6 Add input padding: 8px 12px (currently uses 0.5rem)
- [x] 3.7 Add focus state: border-color changes to '#3b82f6'

## 4. Read-Only Fields (Section 12 of UI_CONVENTIONS)

- [x] 4.1 N/A - Non-editable fields render as spans, not inputs
- [x] 4.2 N/A - Non-editable fields render as spans, not inputs  
- [x] 4.3 Verify "(read-only)" suffix displays in #666

## 5. Additional Alignments

- [x] 5.1 Update button gap from 1rem to 12px (0.75rem)
- [x] 5.2 Verify checkbox alignment with label (gap 8px)
- [x] 5.3 Verify table column styling matches spec
- [x] 5.4 Verify LLM tab styling (active bg: #2a2a2a, inactive color: #888)

## 6. FormModal Button Colors (Section 2, 16 of UI_CONVENTIONS)

- [x] 6.1 Update FormModal submit button: add backgroundColor: '#22c55e', color: '#fff'
- [x] 6.2 Update FormModal Cancel button: add backgroundColor: '#4b5563', color: '#fff'
- [x] 6.3 Add loading state: opacity 0.5 when isSubmitting=true
- [x] 6.4 Verify button dimensions: min-width 80px, height 36px

## 7. Login Button Colors (Section 28 of UI_CONVENTIONS)

- [x] 7.1 Update Login submit button: add backgroundColor: '#22c55e', color: '#fff'
- [x] 7.2 Add loading state: opacity 0.5 when submitting=true, text "Please wait..."
- [x] 7.3 Verify button width: 100% (full width of form)

## 8. Button Colors (Dark Theme Update) - Section 2 Updated

- [x] 8.1 Update Save button: change to #4b5563 (gray)
- [x] 8.2 Update Apply button: change to #4b5563 (gray)
- [x] 8.3 Update Save&Apply button: change to #4b5563 (gray)
- [x] 8.4 Update Delete button: change to #4b5563 (gray)
- [x] 8.5 Update Add buttons across all components (TaskList, PersonaList, etc.)
- [x] 8.6 Update Cancel buttons: keep #4b5563 (already good)
- [x] 8.7 Verify all button colors in all components match new spec

## 9. Icon Visibility Fixes - New Section 2b

- [x] 9.1 ConfigEditor: Fix Eye/EyeSlash icons at lines 383, 385 (add explicit stroke)
- [x] 9.2 ConfigEditor: Fix Eye/EyeSlash icons at lines 573, 575 (provider modal)
- [x] 9.3 ConfigEditor: Fix Eye/EyeSlash icons at lines 769, 771 (provider list)
- [x] 9.4 ConfigEditor: Fix PencilIcon (edit) at lines 783, 862
- [x] 9.5 ConfigEditor: Fix TrashIcon (delete) at lines 790, 869
- [x] 9.6 Add explicit stroke="currentColor" to all heroicons for dark theme visibility
- [x] 9.7 TaskList: Verify edit/delete icons work (line 326)
- [x] 9.8 TaskEditor: Verify delete icon works (line 276)
- [x] 9.9 PersonaEditor: Verify delete icon works (line 274)

## 9b. Icon Button Container Sizing (NEW - Critical Fix)

- [x] 9b.1 ConfigEditor: Fix edit/delete button containers - add width/height 32px, display flex, lighter color #d1d5db
- [x] 9b.2 ConfigEditor: Fix Eye/EyeSlash toggle button containers in forms
- [x] 9b.3 ConfigEditor: Fix Eye/EyeSlash toggle button containers in provider list
- [x] 9b.4 TaskList: Fix PlusIcon, ArrowPathIcon, PencilSquareIcon button containers
- [x] 9b.5 TaskEditor: Fix TrashIcon button container
- [x] 9b.6 PersonaEditor: Fix TrashIcon button container
- [x] 9b.7 Update all icon colors from #4b5563 to #d1d5db (light gray for visibility)
- [x] 9b.8 Verify all icon buttons have 32x32px clickable area
- [x] 9b.9 CRITICAL FIX: Update all icons to use solid (not outline) - spec and code updated

## 10. Button Text Alignment - Section 2 Updated

**NOTE: Tasks 10.1-10.7 deferred to future work**

## 11. Verification (Final)

**NOTE: Tasks 11.1-11.10 deferred to future work**

---

*Change archived: March 22, 2026 - Incomplete tasks removed, main UI compliance work complete*