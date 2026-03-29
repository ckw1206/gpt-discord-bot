import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import Modal from './Modal'
import { ArrowLeft, FileDown, Trash2 } from 'lucide-react'
import { Button } from './ui/button'
import { Input } from './ui/input'

const API_BASE = '/api'

interface PersonaDetail {
  name: string
  filename: string
  content: string
  word_count: number
}

interface PersonaUsage {
  name: string
  filename: string
  type: 'task' | 'model'
  schedule?: string
}

interface PersonaEditorProps {
  token: string
  personaName?: string | null  // null for new persona
  onSave?: () => void
  onSaveWithName?: (name: string) => void  // Called after creating new persona with the name
  onCancel?: () => void
}

// Default template content from persona-example.md
const TEMPLATE_CONTENT = `**Role:** You are a {role_description}.

**Operational Rules (CRITICAL):**
1. **Time Lock:** Confirm today's date \`{date}\`.
   - If it's Monday, {rule_if_monday}.
   - If it's Tuesday‑Saturday, {rule_if_weekday}.
   - **Strictly disallow** referencing non‑trading days or future dates with fabricated data.

2. **Data Retrieval Strategy:**
   - **Step 1:** Call \`{tool_name1}\` to obtain the required data (must be the first call).
   - **Step 2 (optional):** If additional context is needed, use \`{tool_name2}\`.
   - **Prohibit** emitting data or warnings before \`{tool_name1}\` has run.

3. **Discord Format Guidelines:**
   - **No Markdown tables** (avoid \`|--|\`).
   - **Emphasis tags:** Important info must be **bolded**.

4. **Language:** Must use English only.

**Output Structure:**
### 📅 {report_title}: [YYYY‑MM‑DD] ({context_info})

**Summary:** [Two‑sentence overview of key points]

**Data Highlights:** [Relevant data or observations]

**Insights (Optional):**  
- Insight 1  
- Insight 2  
- Insight 3

**Error Handling:**
- If \`{tool}\` returns no data → mark with ⚠️ Unable to retrieve accurate data for [date]; do **not** fabricate numbers.

**Current Context:**
- Today is \`{date}\`, Location is \`{location}\`.
`

export default function PersonaEditor({ token, personaName, onSave, onSaveWithName, onCancel }: PersonaEditorProps) {
  const [name, setName] = useState(personaName || '')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(!!personaName)
  const [saving, setSaving] = useState(false)
  const [showPreview, setShowPreview] = useState(true)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [usage, setUsage] = useState<PersonaUsage[]>([])

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Load existing persona if editing
  useEffect(() => {
    if (personaName) {
      fetchPersona(personaName)
    }
  }, [personaName])

  const fetchPersona = async (name: string) => {
    setLoading(true)
    try {
      const res = await axios.get<PersonaDetail>(`${API_BASE}/personas/${name}`, authHeaders)
      setName(res.data.name)
      setContent(res.data.content)
      
      // Fetch usage info (21.5.3)
      try {
        const usageRes = await axios.get<PersonaUsage[]>(`${API_BASE}/personas/${name}/usage`, authHeaders)
        setUsage(usageRes.data)
      } catch {
        // Usage endpoint may not exist yet
        setUsage([])
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to load persona')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error('Persona name is required')
      return
    }

    // Validate name format
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      toast.error('Name can only contain letters, numbers, dash, and underscore')
      return
    }

    setSaving(true)

    try {
      const isNew = !personaName
      if (isNew) {
        await axios.post(`${API_BASE}/personas`, { name, content }, authHeaders)
        toast.success(`Persona "${name}" created successfully`)
        // Set the name so we switch from "new" mode to "edit" mode
        // The parent will update editingPersona state via onSaveWithName
        if (onSaveWithName) {
          onSaveWithName(name)
        }
      } else {
        await axios.put(`${API_BASE}/personas/${name}`, { content }, authHeaders)
        toast.success(`Persona "${name}" saved successfully`)
      }

      // Note: Apply/Refresh not needed - bot reads persona from disk each time
      // So we stay in editor mode instead of navigating back
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save persona')
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteClick = () => {
    if (!personaName) return
    setShowDeleteModal(true)
  }

  const handleDeleteConfirm = async () => {
    if (!personaName) return

    setSaving(true)

    try {
      await axios.delete(`${API_BASE}/personas/${personaName}`, authHeaders)
      toast.success(`Persona "${personaName}" deleted`)
      onSave?.()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to delete persona')
    } finally {
      setSaving(false)
    }
  }

  const loadTemplate = () => {
    setContent(TEMPLATE_CONTENT)
  }

  if (loading) {
    return <div className="p-8">Loading persona...</div>
  }

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">{personaName ? `Edit: ${personaName}` : 'New Persona'}</h2>
        <Button variant="ghost" onClick={onCancel}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>
      </div>

      {/* Usage info - enhanced to show tasks AND models */}
      {usage.length > 0 && (
        <div className="p-3 bg-amber-950 border border-amber-600 rounded text-sm">
          <strong className="text-amber-500">⚠️ Used by {usage.length} item(s):</strong>
          
          {/* Separate tasks and models */}
          {usage.filter(u => u.type === 'task').length > 0 && (
            <>
              <div className="mt-2 font-bold">Tasks:</div>
              <ul className="mt-1 pl-6">
                {usage.filter(u => u.type === 'task').map((u) => (
                  <li key={u.filename}>
                    <code>{u.name}</code>
                    {u.schedule && <span className="text-muted-foreground ml-2">schedule: {u.schedule}</span>}
                  </li>
                ))}
              </ul>
            </>
          )}
          
          {usage.filter(u => u.type === 'model').length > 0 && (
            <>
              <div className="mt-2 font-bold">Models (config.yaml):</div>
              <ul className="mt-1 pl-6">
                {usage.filter(u => u.type === 'model').map((u) => (
                  <li key={u.filename}>
                    <code>{u.name}</code>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {/* Name input for new personas */}
      {!personaName && (
        <div>
          <label className="block mb-2">Persona Name:</label>
          <Input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-persona"
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 flex-wrap">
        <Button onClick={handleSave} disabled={saving}>
          Save
        </Button>
        {!personaName && (
          <Button variant="outline" onClick={loadTemplate} disabled={saving}>
            <FileDown className="w-4 h-4 mr-2" />
            Use Template
          </Button>
        )}
        {personaName && !personaName.endsWith('-example') && (
          <Button
            variant="destructive"
            onClick={handleDeleteClick}
            disabled={saving}
            className="ml-auto"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Delete
          </Button>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && personaName && (
        <Modal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          onConfirm={handleDeleteConfirm}
          title="Delete Persona"
          message={`Are you sure you want to delete "${personaName}"? This action cannot be undone.`}
          confirmText="Delete"
          cancelText="Cancel"
          variant="danger"
        />
      )}

      {/* Editor + Preview */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Editor */}
        <div className="flex-1 flex flex-col">
          <div className="mb-2 flex justify-between">
            <span>Markdown Editor</span>
            <Button variant="outline" size="sm" onClick={() => setShowPreview(!showPreview)}>
              {showPreview ? 'Hide' : 'Show'} Preview
            </Button>
          </div>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Write your persona content in Markdown..."
            className="flex-1 resize-none p-4 bg-background border rounded text-foreground font-mono text-sm leading-6"
          />
        </div>

        {/* Preview */}
        {showPreview && (
          <div className="flex-1 flex flex-col">
            <span className="mb-2">Preview</span>
            <div
              className="flex-1 overflow-auto p-4 bg-background border rounded text-foreground text-sm leading-6"
            >
              <ReactMarkdown>{content || '*No content*'}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {/* Word count */}
      <div className="text-muted-foreground text-sm">
        {content.split(/\s+/).filter(Boolean).length} words
      </div>
    </div>
  )
}