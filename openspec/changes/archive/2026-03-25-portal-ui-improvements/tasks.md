# Portal UI Improvements - Task Status Summary

## Overview

| Section | Status | Notes |
|---------|--------|-------|
| 1. Task Split-View | ✅ Complete | Fixed mobile overlap - TaskList hides when editor active |
| 2. Tool Name Alignment | ✅ Complete | Working - shows actual tool names |
| 3. Token Verification | ✅ Complete | All tasks done |
| 4. Unsaved Changes | ✅ Complete | Working - dialog appears correctly |
| 5. Config Button Styling | ✅ Complete | Using shadcn/ui Button components |
| 6. Uptime Counter | ✅ Complete | Working - increments every minute |
| 7. Avatar Centering | ✅ Complete | Working - emoji centered |
| 8. Mood→Status | ✅ Complete | Working - shows "Status" |
| 9. Lucide Icons | ✅ Complete | Replaced emoji with Lucide icons |
| 10. Title Consistency | ✅ Complete | All titles standardized |
| 11. Refresh Button | ✅ Complete | Working - consistent across pages |
| 12. Verification & Testing | ✅ Complete | Test scenarios documented |
| 13. Task Dashboard | ✅ Complete | Auto-load, hide edit, unsaved warning all implemented |

---

## 1. Task Split-View Interface

Refactor task management from full-page navigation to split-view panel with structured form fields.

### 1.1 Dashboard Layout

- [x] 1.1.1 Replace conditional rendering with split-view flexbox layout
- [x] 1.1.2 Always render both TaskList (left) and TaskEditor (right)
- [x] 1.1.3 Add selectedTask state for panel mode
- [x] 1.1.4 Responsive layout: stack on mobile (<768px), split on desktop

### 1.2 TaskList Component Updates

- [x] 1.2.1 Add selectedTask prop for highlighting active card
- [x] 1.2.2 Add visual indicator (border/color) for selected card
- [x] 1.2.3 Update "Add New" button to open editor in right panel (no navigation)
- [x] 1.2.4 Remove full-page mode - always run in panel

### 1.3 TaskEditor Panel Mode

- [x] 1.3.1 Convert from Modal to side panel layout
- [x] 1.3.2 Add structured form fields instead of raw YAML
- [x] 1.3.3 Add Save/Cancel buttons in editor panel
- [x] 1.3.4 Add close button (X) to dismiss editor

### 1.4 Structured Form Fields

- [x] 1.4.1 Name field (text input, required, unique validation)
- [x] 1.4.2 Cron field (text input, cron validation)
- [x] 1.4.3 Enabled field (checkbox)
- [x] 1.4.4 Model field (text input with suggestions from config)
- [x] 1.4.5 Persona field (text input with suggestions)
- [x] 1.4.6 Prompt field (textarea, required)
- [x] 1.4.7 Tools field (multi-select checklist from tools.yaml)
- [x] 1.4.8 User ID field (text input, preserve as string for precision)
- [x] 1.4.9 Channel ID field (text input, preserve as string for precision)

### 1.5 Validation Features

- [x] 1.5.1 Client-side cron expression validation
- [x] 1.5.2 Show validation error for invalid cron format
- [x] 1.5.3 Required field validation before save
- [x] 1.5.4 Unique name validation (no duplicate tasks)

### 1.6 Responsive Behavior

- [x] 1.6.1 Desktop (≥1024px): Two-column split view (50%/50%)
- [x] 1.6.2 Tablet (768-1023px): Two-column (40%/60%)
- [x] 1.6.3 Mobile (<768px): Stacked layout with full-screen editor overlay
- [x] 1.6.4 Back button on mobile to return to list

> **Fixed:** On mobile, TaskList now hides when TaskEditor is active (using `hidden md:flex` classes). TaskEditor shows "Close" button on desktop, "X" icon on mobile.

### 1.7 Backend Integration

- [x] 1.7.1 Reuse existing PUT /api/tasks/{name} endpoint
- [x] 1.7.2 Reuse existing POST /api/tasks for new tasks
- [x] 1.7.3 Preserve config structure (YAML backend unchanged)
- [x] 1.7.4 Handle save success/error feedback

## 2. Tool Name Alignment

Fix the TaskEditor tool dropdown to show actual tool names instead of skill file names.

### Problem
The TaskEditor's tool multi-select fetches from `/api/skills` which returns skill file names 
(e.g., "yahoo_finance") instead of actual tool names (e.g., "get_market_prices"). This creates 
a mismatch when users select tools that don't exist in the registry.

### 2.1 Backend API Changes

- [x] 2.1.1 Create `/api/tools` endpoint that returns tool names from registry
- [x] 2.1.2 Include tool schema (name, description, parameters) in response

### 2.2 Frontend Integration

- [x] 2.2.1 Update TaskEditor to fetch from `/api/tools` instead of `/api/skills`
- [x] 2.2.2 Update tool dropdown to display actual tool names
- [x] 2.2.3 Verify selected tools work with execution layer

### 2.3 Testing

- [x] 2.3.1 Verify tool names match registry (get_market_prices, web_search, visuals_core)
- [x] 2.3.2 Test that selected tools execute correctly
- [x] 2.3.3 Verify backward compatibility (existing tasks still work)

## 3. Token Verification Fix

Fix the web portal to properly handle authentication after database reset. When the database is deleted/refreshed, the old JWT token should be invalidated and users should be redirected to the setup page.

### Problem
When the database is deleted/refreshed, the old JWT token remains valid in the browser, allowing continued access to the dashboard instead of redirecting to the setup page.

### 3.1 Backend Changes

- [x] 3.1.1 Add `token_version` field to User model (default=0)
- [x] 3.1.2 Include `token_version` in JWT token creation (setup and login)
- [x] 3.1.3 Verify `token_version` in `get_current_user` auth dependency
- [x] 3.1.4 Create `/api/auth/me` endpoint for token verification

### 3.2 Frontend Changes

- [x] 3.2.1 Add token verification on app startup (call `/api/auth/me`)
- [x] 3.2.2 Clear token from localStorage on 401 response
- [x] 3.2.3 Redirect to login when token is invalid
- [x] 3.2.4 Fix React hooks rule violation (useEffect before conditional return)

### 3.3 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Flow: Token Verification on Page Load                     │
├─────────────────────────────────────────────────────────────┤
│  1. App loads → Read token from localStorage                │
│  2. Call /api/auth/me with token                            │
│     ├─ 200 OK → Token valid, show dashboard                │
│     └─ 401 → Token invalid:                                │
│         - Clear localStorage                               │
│         - Redirect to /login                               │
│  3. Login page → Call /api/auth/has-users                   │
│     ├─ has_users=false → Show setup page                   │
│     └─ has_users=true → Show login form                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Token Version Logic

| Scenario | Old Token Version | New User Version | Result |
|----------|------------------|-------------------|--------|
| After fix (default=0) | 1 | 0 | ❌ Invalid (correct) |
| Normal login | 0 | 0 | ✅ Valid |

### 3.5 Files Modified

| File | Change |
|------|--------|
| `bot/db/models.py` | Add `token_version` field |
| `bot/web/auth.py` | Add version check in `get_current_user` |
| `bot/web/server.py` | Add `/api/auth/me` endpoint |
| `web/src/App.tsx` | Add token verification on startup |

### 3.6 Verification

- [x] 3.6.1 Backend: Token with version=1 rejected after DB reset
- [x] 3.6.2 Backend: New token with version=0 works normally
- [x] 3.6.3 Frontend: Hard refresh redirects to login/setup after DB reset
- [x] 3.6.4 Frontend: No React errors in console

## 4. Unsaved Changes Protection

Add confirmation dialog when clicking "Add New" button while editing a task that has unsaved changes.

### Problem
When a user is editing a task and clicks "Add New" to create a new task, the current unsaved changes are lost without warning.

### 4.1 TaskEditor Changes

- [x] 4.1.1 Add `isDirty` state to track form changes
- [x] 4.1.2 Track changes in all form fields (name, cron, enabled, model, persona, prompt, tools, userId, channelId)
- [x] 4.1.3 Expose `isDirty` value via prop or callback

### 4.2 Dashboard Integration

- [x] 4.2.1 Pass `onHasUnsavedChanges` callback from TaskEditor to Dashboard
- [x] 4.2.2 Add state to track if TaskEditor has unsaved changes
- [x] 4.2.3 Show confirmation dialog when `onRequestCreate` is called with unsaved changes

### 4.3 Dialog Component

- [x] 4.3.1 Use shadcn/ui Dialog component for confirmation
- [x] 4.3.2 Display warning message about unsaved changes
- [x] 4.3.3 Provide "Save & Continue", "Discard", and "Cancel" options

### 4.4 Files Modified

| File | Change |
|------|--------|
| `web/src/components/TaskEditor.tsx` | Add isDirty state and tracking |
| `web/src/components/Dashboard.tsx` | Add confirmation dialog logic |
| `web/src/components/TaskList.tsx` | Pass unsaved changes callback |

## 5. Config Dashboard Button Styling

Replace raw HTML buttons with shadcn/ui Button components in Config dashboard.

### Problem
The Config dashboard uses raw `<button>` elements with inline styles instead of shadcn/ui Button components, causing inconsistent UI with other dashboards.

### 5.1 Button Replacement

- [x] 5.1.1 Replace Save button with `<Button variant="secondary">`
- [x] 5.1.2 Replace Apply button with `<Button variant="outline">`
- [x] 5.1.3 Replace Save&Apply button with `<Button variant="default">`
- [x] 5.1.4 Remove inline styles from all buttons

> **Status: COMPLETE** - Using shadcn/ui Button components

### 5.2 Files Modified

| File | Change |
|------|--------|
| `web/src/components/ConfigEditor.tsx` | Replace raw buttons with shadcn/ui Button |

## 6. Uptime Counter Enhancement

Implement local counter that continues incrementing after receiving initial data from API.

### Problem
The uptime value is fetched once on load and never updates, showing a stale value.

### 6.1 Implementation

- [x] 6.1.1 Add `localUptime` state in Dashboard
- [x] 6.1.2 Initialize from API response `uptime_seconds`
- [x] 6.1.3 Add setInterval to increment every 60 seconds
- [x] 6.1.4 Add periodic API refresh (every 5 minutes) to stay in sync
- [x] 6.1.5 Clean up interval on unmount

### 6.2 Files Modified

| File | Change |
|------|--------|
| `web/src/components/Dashboard.tsx` | Add local uptime counter |

## 7. Avatar Centering Fix

Fix robot emoji centering in avatar placeholder.

### Problem
The avatar placeholder uses invalid CSS property `justify-content` instead of `justify-center`.

### 7.1 Implementation

- [x] 7.1.1 Change `justify-content` to `justify-center` in avatar div
- [x] 7.1.2 Verify centering works correctly

### 7.2 Files Modified

| File | Change |
|------|--------|
| `web/src/components/Dashboard.tsx` | Fix CSS class |

## 8. Label Rename: Mood to Status

Change "Mood" label to "Status" in dashboard.

### Problem
The label "Mood" is unclear - "Status" is more descriptive for the bot's status message.

### 8.1 Implementation

- [x] 8.1.1 Change `<strong>Mood:</strong>` to `<strong>Status:</strong>` in Dashboard.tsx

### 8.2 Files Modified

| File | Change |
|------|--------|
| `web/src/components/Dashboard.tsx` | Update label text |

## 9. Emoji to Lucide Icons

Replace emoji icons with Lucide icons in Config dashboard section navigation.

### Problem
Config dashboard uses emoji (💬, ⚙️, etc.) while other dashboards use Lucide icons, creating inconsistency.

### 9.1 Implementation

- [x] 9.1.1 Import Lucide icons (MessageSquare, Settings, Database, Cpu, Mic, Globe)
- [x] 9.1.2 Replace emoji in CONFIG_SECTIONS with icon components
- [x] 9.1.3 Update rendering to use icon component

> **Status: COMPLETE** - Replaced emoji with Lucide icons using React.createElement()

### 9.2 Files Modified

| File | Change |
|------|--------|
| `web/src/components/ConfigEditor.tsx` | Replace emoji with Lucide icons |

## 10. Dashboard Title Consistency

Standardize all dashboard titles to use consistent styling.

### Problem
Different dashboards have different title styles (font size, icon size, spacing).

### 10.1 Standard Title Pattern

All dashboard titles should follow this pattern:
```tsx
<h2 className="flex items-center gap-2 text-xl font-semibold">
  <Icon className="w-6 h-6" />
  Title
</h2>
```

### 10.2 Implementation

- [x] 10.2.1 Update Config dashboard title - **APPLIED**
- [x] 10.2.2 Update ServerList title - **APPLIED**
- [x] 10.2.3 Update PersonaList title - **APPLIED**
- [x] 10.2.4 Update SkillsList title - **APPLIED**
- [x] 10.2.5 Verify all titles match Dashboard pattern - **COMPLETE**

### 10.3 Files Modified

| File | Change | Status |
|------|--------|--------|
| `web/src/components/ConfigEditor.tsx` | Standardize title | Applied ✓ |
| `web/src/components/ServerList.tsx` | Standardize title | Applied ✓ |
| `web/src/components/PersonaList.tsx` | Standardize title | Applied ✓ |
| `web/src/components/SkillsList.tsx` | Standardize title | Applied ✓ |

## 11. Refresh Button Consistency

Ensure Refresh buttons across all dashboards use consistent styling.

### Problem
TaskList uses different button styling than Dashboard for Refresh.

### 11.1 Implementation

- [x] 11.1.1 Verify TaskList Refresh button matches Dashboard style
- [x] 11.1.2 Update if needed to use `<Button variant="outline" size="sm">`

### 11.2 Files Modified

| File | Change |
|------|--------|
| `web/src/components/TaskList.tsx` | Verify/fix button styling |

---

## 12. Verification & Testing

This section defines the testing approach and acceptance criteria for all UI improvements.

### 12.1 Testing Strategy

**Manual Testing (Priority 1):**
- Test each dashboard page manually in browser
- Verify button interactions work correctly
- Check responsive behavior on different screen sizes

**Visual Regression (Priority 2):**
- Compare screenshots before/after changes
- Verify consistent styling across all pages

**Integration Testing (Priority 3):**
- Test API endpoints return correct data
- Verify form validation works correctly

### 12.2 Test Scenarios

#### Task Management (Section 1-2)
- [x] TaskList displays all tasks correctly
- [x] Clicking a task opens TaskEditor in right panel
- [x] "Add New" button creates new task in right panel
- [x] TaskEditor form fields populate correctly
- [x] Save button creates/updates task successfully
- [x] Delete button removes task after confirmation
- [x] Cron validation shows error for invalid expressions
- [x] Tool dropdown shows actual tool names from registry

#### Unsaved Changes (Section 4)
- [x] Editing a field marks form as dirty
- [x] Clicking "Add New" with unsaved changes shows dialog
- [x] "Save & Continue" saves and creates new task
- [x] "Discard" creates new task without saving
- [x] "Cancel" returns to editing current task

#### YAML View (Recent Fix)
- [x] Open existing task (e.g., stock-market-checker)
- [x] Toggle to "YAML View"
- [x] channel_id shows correctly in YAML view

#### Config Dashboard (Section 5, 9)
- [x] Save button uses shadcn/ui Button styling
- [x] Apply button uses shadcn/ui Button styling
- [x] Save&Apply button uses shadcn/ui Button styling
- [x] Section navigation uses Lucide icons (not emoji)
- [x] All buttons are properly aligned

#### Dashboard (Section 6-8, 10-11)
- [x] Uptime counter increments every minute
- [x] Avatar placeholder shows centered robot emoji
- [x] "Status" label displays instead of "Mood"
- [x] All dashboard titles have consistent styling
- [x] Refresh buttons use consistent styling across pages

### 12.3 Browser Testing Matrix

| Browser | Task Split-View | Config Buttons | Uptime Counter | Responsive |
|---------|-----------------|----------------|----------------|------------|
| Chrome  | ✓ | ✓ | ✓ | ✓ |
| Firefox | ✓ | ✓ | ✓ | ✓ |
| Safari  | ✓ | ✓ | ✓ | ✓ |
| Edge    | ✓ | ✓ | ✓ | ✓ |

### 12.4 Device Testing Matrix

| Device | Task Split-View | Config Dashboard | Dashboard |
|--------|-----------------|------------------|-----------|
| Desktop (1920x1080) | Split view | ✓ | ✓ |
| Tablet (768x1024) | Split view (60/40) | ✓ | ✓ |
| Mobile (375x667) | Stacked | ✓ | ✓ |

---

## 13. Task Dashboard Improvements

Improve task card interaction and integrate with unsaved changes protection.

### Problem
1. When expanding a task card, users must click "replace_string_in_file" button again to open TaskEditor - redundant step
2. replace_string_in_file button shows in expanded card even when TaskEditor is already visible - confusing UX
3. Clicking other task cards during editing doesn't show unsaved warning - can lose changes

### 13.1 Auto-load Task on Expand

- [x] 13.1.1 Add `onExpand` callback prop to TaskList component
- [x] 13.1.2 Call `onExpand(taskName)` when task card is expanded
- [x] 13.1.3 Update Dashboard to set `editingTask` when `onExpand` is called

### 13.2 Hide replace_string_in_file Button When Expanded

- [x] 13.2.1 Conditionally render replace_string_in_file button only when task is NOT expanded
- [x] 13.2.2 Keep "Run Now" button visible in expanded view

### 13.3 Unsaved Warning for Task Clicks

- [x] 13.3.1 Add `onRequestExpand` callback prop to TaskList component
- [x] 13.3.2 Call `onRequestExpand(taskName)` instead of directly expanding
- [x] 13.3.3 Update Dashboard to check `taskHasUnsavedChanges` before expanding
- [x] 13.3.4 Show unsaved dialog if changes exist, proceed with expand after confirmation

### 13.4 Files Modified

| File | Change |
|------|--------|
| `web/src/components/TaskList.tsx` | Add onExpand/onRequestExpand props, update click handlers, conditionally hide edit button |
| `web/src/components/Dashboard.tsx` | Add handlers for task expand, integrate with existing unsaved changes dialog |

### 13.5 Implementation Notes

- Reuse existing `handleRequestCreateTask` pattern for unsaved changes dialog
- The `pendingAction` callback pattern already exists in Dashboard - reuse it
- TaskEditor already loads task data when `taskName` prop changes - no additional loading logic needed

### 12.5 Acceptance Criteria

**Must Pass (P0):**
- [x] Task split-view works on desktop
- [x] Config Save/Apply buttons function correctly
- [x] No console errors on any dashboard
- [x] All buttons are clickable and responsive

**Should Pass (P1):**
- [x] Unsaved changes dialog appears correctly
- [x] Uptime counter increments properly
- [x] Avatar is properly centered
- [x] All titles have consistent styling

**Nice to Have (P2):**
- [x] Emoji replaced with Lucide icons
- [x] "Mood" renamed to "Status"
- [x]] Refresh buttons consistent across all pages

### 12.6 Bug Reporting Template

When reporting issues, include:
```
**Environment:**
- Browser: [Chrome/Firefox/Safari/Edge]
- Device: [Desktop/Tablet/Mobile]
- Screen size: [width x height]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Screenshots:**
[Attach screenshots if applicable]
```