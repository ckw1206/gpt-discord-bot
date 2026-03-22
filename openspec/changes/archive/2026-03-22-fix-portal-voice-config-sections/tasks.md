## 1. Backend Changes

- [x] 1.1 Update EDITABLE_FIELDS in bot/web/routes/config.py to add voice fields (voice.region, voice.default_voice)
- [x] 1.2 Update EDITABLE_FIELDS to add nested portal fields (portal.enabled, portal.port, portal.logs.retention_days, portal.logs.levels, portal.cors_origins)
- [x] 1.3 Update read_only_fields to use "voice" instead of "azure-speech"

## 2. Frontend - CONFIG_SECTIONS Updates

- [x] 2.1 Update CONFIG_SECTIONS in web/src/components/ConfigEditor.tsx to add nested portal fields
- [x] 2.2 Update CONFIG_SECTIONS to add voice fields (voice.region, voice.default_voice)

## 3. Frontend - Field Extraction Logic

- [x] 3.1 Fix fetchConfig to extract portal fields from apiData.portal (separate from config)
- [x] 3.2 Fix field extraction to properly match voice fields from apiData.config.voice
- [x] 3.3 Add proper handling for nested key matching (e.g., portal.logs.retention_days)

## 4. Frontend - Input Rendering

- [x] 4.1 Add boolean field rendering for portal.enabled toggle
- [x] 4.2 Add array field rendering for portal.logs.levels (comma-separated input)
- [x] 4.3 Add array field rendering for portal.cors_origins (comma-separated input)

## 5. Testing

- [x] 5.1 Build frontend and verify no errors

### Manual Testing Required (run web portal and bot)

- [x] 5.2 Verify Portal section appears in ConfigEditor sidebar
- [x] 5.3 Verify Voice section appears in ConfigEditor sidebar
- [x] 5.4 Test saving portal.enabled toggle
- [x] 5.5 Test saving portal.port value
- [x] 5.6 Test saving voice.region value
- [x] 5.7 Verify changes persist to config.yaml