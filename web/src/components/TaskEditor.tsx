import { useState, useEffect } from 'react'
import axios from 'axios'
import yaml from 'js-yaml'
import { toast } from 'react-hot-toast'
import { Trash2, X, AlertCircle } from 'lucide-react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Checkbox } from './ui/checkbox'
import { ScrollArea } from './ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog'

const API_BASE = '/api'

// Cron validation regex - accepts any 5 space-separated fields (allows day names like MON-FRI)
const CRON_REGEX = /^\S+\s+\S+\s+\S+\s+\S+\s+\S+$/

interface TaskDetail {
  name: string
  filename: string
  config: Record<string, any>
  status: string
}

interface TaskEditorProps {
  token: string
  taskName?: string | null  // null for new task
  onSave?: () => void
  onSaveWithName?: (name: string) => void  // Called after creating new task with the name
  onCancel?: () => void
  onDirtyChange?: (isDirty: boolean) => void  // Callback for unsaved changes (Task 4.1.3)
}

// Default template for new tasks
const TEMPLATE_CONFIG = {
  name: 'new-task',
  enabled: true,
  cron: '0 * * * *',
  user_id: '',
  channel_id: '',
  model: 'google/gemini-2.5-flash',
  prompt: 'Your task prompt here',
  tools: [],
  persona: ''
}

// Available models (common models used in the bot)
const AVAILABLE_MODELS = [
  'google/gemini-2.5-flash',
  'google/gemini-2.0-flash',
  'openai/gpt-4o-mini',
  'openai/gpt-4o',
  'anthropic/claude-3-5-sonnet-20241022',
  'anthropic/claude-3-haiku-20240307',
  'mistralai/mistral-small-250111',
  'meta-llama/llama-3.1-70b-instruct',
]

export default function TaskEditor({ token, taskName, onSave, onSaveWithName, onCancel, onDirtyChange }: TaskEditorProps) {
  // Form state - structured fields
  const [name, setName] = useState('')
  const [cron, setCron] = useState('0 * * * *')
  const [enabled, setEnabled] = useState(true)
  const [model, setModel] = useState('google/gemini-2.5-flash')
  const [persona, setPersona] = useState('')
  const [prompt, setPrompt] = useState('')
  const [tools, setTools] = useState<string[]>([])
  const [userId, setUserId] = useState('')
  const [channelId, setChannelId] = useState('')
  
  // UI state
  const [originalName, setOriginalName] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [showYamlView, setShowYamlView] = useState(false)
  const [configText, setConfigText] = useState('')
  const [validationErrors, setValidationErrors] = useState<string[]>([])
  
  // Track unsaved changes (Task 4.1.1)
  // Note: isDirty is used via onDirtyChange callback, not directly read
  const [_isDirty, setIsDirty] = useState(false)
  
  // Available options (loaded from config)
  const [availableTools, setAvailableTools] = useState<string[]>([])
  const [availablePersonas, setAvailablePersonas] = useState<string[]>([])

  // Track original values for dirty checking (Task 4.1.2)
  const [originalValues, setOriginalValues] = useState({
    name: '',
    cron: '0 * * * *',
    enabled: true,
    model: 'google/gemini-2.5-flash',
    persona: '',
    prompt: '',
    tools: [] as string[],
    userId: '',
    channelId: '',
  })

  // Update isDirty when form values change (Task 4.1.2)
  useEffect(() => {
    // Don't update dirty state while loading - wait for data to be ready
    if (loading) return
    
    const currentValues = { name, cron, enabled, model, persona, prompt, tools, userId, channelId }
    const hasChanges = JSON.stringify(currentValues) !== JSON.stringify(originalValues)
    setIsDirty(hasChanges)
    // Notify parent component of dirty state change (Task 4.1.3)
    onDirtyChange?.(hasChanges)
  }, [name, cron, enabled, model, persona, prompt, tools, userId, channelId, originalValues, onDirtyChange, loading])

  // Update YAML view whenever form fields change (after loading is complete)
  useEffect(() => {
    if (!loading) {
      updateYamlFromForm()
    }
  }, [name, cron, enabled, model, persona, prompt, tools, userId, channelId, loading])

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Load available tools and personas on mount
  useEffect(() => {
    fetchTools()
    fetchPersonas()
  }, [])

  const fetchTools = async () => {
    try {
      // Use /tools endpoint to get actual tool names from registry
      const res = await axios.get(`${API_BASE}/tools`, authHeaders)
      if (res.data && Array.isArray(res.data)) {
        setAvailableTools(res.data.map((t: any) => t.name))
      }
    } catch (err) {
      console.error('Failed to fetch tools:', err)
      // Fallback to common tools from registry
      setAvailableTools(['web_search', 'visuals_core', 'get_market_prices', 'google_tools'])
    }
  }

  const fetchPersonas = async () => {
    try {
      const res = await axios.get(`${API_BASE}/personas`, authHeaders)
      if (res.data && Array.isArray(res.data)) {
        setAvailablePersonas(res.data.map((p: any) => p.name))
      }
    } catch (err) {
      console.error('Failed to fetch personas:', err)
    }
  }

  // Load existing task if editing
  useEffect(() => {
    if (taskName) {
      fetchTask(taskName)
    } else if (taskName === undefined || taskName === null || taskName === '') {
      // New task - always use template (reset form for new task)
      loadTemplate()
    }
  }, [taskName])

  const loadTemplate = () => {
    setName(TEMPLATE_CONFIG.name)
    setCron(TEMPLATE_CONFIG.cron)
    setEnabled(TEMPLATE_CONFIG.enabled)
    setModel(TEMPLATE_CONFIG.model)
    setPersona(TEMPLATE_CONFIG.persona)
    setPrompt(TEMPLATE_CONFIG.prompt)
    setTools(TEMPLATE_CONFIG.tools)
    setUserId(String(TEMPLATE_CONFIG.user_id || ''))
    setChannelId(String(TEMPLATE_CONFIG.channel_id || ''))
    setOriginalName('')
    // Store original values for dirty checking (Task 4.1.2)
    setOriginalValues({
      name: TEMPLATE_CONFIG.name,
      cron: TEMPLATE_CONFIG.cron,
      enabled: TEMPLATE_CONFIG.enabled,
      model: TEMPLATE_CONFIG.model,
      persona: TEMPLATE_CONFIG.persona,
      prompt: TEMPLATE_CONFIG.prompt,
      tools: TEMPLATE_CONFIG.tools,
      userId: String(TEMPLATE_CONFIG.user_id),
      channelId: String(TEMPLATE_CONFIG.channel_id),
    })
    setLoading(false)
    updateYamlFromForm()
  }

  const fetchTask = async (taskName: string) => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/tasks/${taskName}`, authHeaders)
      const task: TaskDetail = res.data
      const config = task.config
      
      // Use task.name (filename from API) as the canonical name, not config.name
      // This ensures delete/update operations use the correct filename
      setName(task.name || '')
      setOriginalName(task.name || '')
      setCron(config.cron || '0 * * * *')
      setEnabled(config.enabled !== false)
      setModel(config.model || 'google/gemini-2.5-flash')
      setPersona(config.persona || '')
      setPrompt(config.prompt || '')
      setTools(config.tools || [])
      
      // Store original values for dirty checking (Task 4.1.2)
      setOriginalValues({
        name: config.name || task.name || '',
        cron: config.cron || '0 * * * *',
        enabled: config.enabled !== false,
        model: config.model || 'google/gemini-2.5-flash',
        persona: config.persona || '',
        prompt: config.prompt || '',
        tools: config.tools || [],
        userId: String(config.user_id || ''),
        channelId: String(config.channel_id || ''),
      })
      
      // Handle ID fields - always use separate user_id and channel_id
      setUserId(String(config.user_id || ''))
      setChannelId(String(config.channel_id || ''))
      
      updateYamlFromForm()
    } catch (err: any) {
      console.error('Failed to fetch task:', err)
      toast.error(err.response?.data?.detail || 'Failed to load task')
    } finally {
      setLoading(false)
    }
  }

  // Build config object from form fields
  const buildConfig = (): Record<string, any> => {
    const config: Record<string, any> = {
      name,
      enabled,
      cron,
      model,
      prompt,
    }
    
    // Add persona if specified
    if (persona) {
      config.persona = persona
    }
    
    // Add tools if specified
    if (tools.length > 0) {
      config.tools = tools
    }
    
    // Add ID fields - always separate user_id and channel_id
    if (userId) config.user_id = userId
    if (channelId) config.channel_id = channelId
    
    return config
  }

  // Update YAML text from form (for advanced view)
  // Note: We pass values directly to avoid React state async issues
  const updateYamlFromForm = () => {
    const config: Record<string, any> = {
      name,
      enabled,
      cron,
      model,
      prompt,
    }
    
    if (persona) config.persona = persona
    if (tools.length > 0) config.tools = tools
    if (userId) config.user_id = userId
    if (channelId) config.channel_id = channelId
    
    setConfigText(yaml.dump(config, { indent: 2, lineWidth: -1 }))
  }

  // Validate form fields
  const validateForm = (): string[] => {
    const errors: string[] = []
    
    // Required fields
    if (!name || name.trim() === '') {
      errors.push('Name is required')
    }
    
    // Name format validation
    if (name && !/^[a-z0-9-]+$/.test(name)) {
      errors.push('Name can only contain lowercase letters, numbers, and hyphens')
    }
    
    // Cron validation
    if (!cron || !CRON_REGEX.test(cron)) {
      errors.push('Invalid cron expression (use format: minute hour day month weekday)')
    }
    
    // Prompt required
    if (!prompt || prompt.trim() === '') {
      errors.push('Prompt is required')
    }
    
    // At least one ID field required
    if (!userId && !channelId) {
      errors.push('Either User ID or Channel ID is required')
    }
    
    return errors
  }

  const handleSave = async () => {
    const errors = validateForm()
    if (errors.length > 0) {
      setValidationErrors(errors)
      errors.forEach(err => toast.error(err))
      return
    }
    setValidationErrors([])

    const config = buildConfig()
    const isNewTask = !originalName

    try {
      setSaving(true)
      
      if (originalName && originalName !== name) {
        // Name changed - create new task first, then delete old
        await axios.post(`${API_BASE}/tasks`, { name, config }, authHeaders)
        await axios.delete(`${API_BASE}/tasks/${originalName}`, authHeaders)
        toast.success('Task renamed!')
        onSaveWithName?.(name)
      } else if (originalName && originalName === name) {
        // Update existing
        await axios.put(`${API_BASE}/tasks/${name}`, { config }, authHeaders)
        toast.success('Task saved!')
      } else {
        // Create new
        await axios.post(`${API_BASE}/tasks`, { name, config }, authHeaders)
        toast.success('Task created!')
        onSaveWithName?.(name)
      }
      
      // Reload task in scheduler (skip for new tasks)
      if (!isNewTask) {
        try {
          await axios.post(`${API_BASE}/tasks/${name}/reload`, {}, authHeaders)
        } catch (reloadErr) {
          console.error('Failed to reload task:', reloadErr)
        }
      }
      
      // Refresh the task data
      fetchTask(name)
    } catch (err: any) {
      console.error('Failed to save task:', err)
      toast.error(err.response?.data?.detail || 'Failed to save task')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!originalName) return

    try {
      setSaving(true)
      await axios.delete(`${API_BASE}/tasks/${originalName}`, authHeaders)
      toast.success('Task deleted!')
      onSave?.()
    } catch (err: any) {
      console.error('Failed to delete task:', err)
      toast.error(err.response?.data?.detail || 'Failed to delete task')
    } finally {
      setSaving(false)
      setShowDeleteModal(false)
    }
  }

  const handleNameChange = (newName: string) => {
    const sanitized = newName.toLowerCase().replace(/[^a-z0-9-]/g, '-')
    setName(sanitized)
    updateYamlFromForm()
  }

  const handleToolToggle = (tool: string) => {
    setTools(prev => {
      if (prev.includes(tool)) {
        return prev.filter(t => t !== tool)
      }
      return [...prev, tool]
    })
    updateYamlFromForm()
  }

  // Show loading or empty state
  if (loading) {
    return <div className="p-8">Loading task...</div>
  }

  const isNewTask = !originalName

  return (
    <div className="flex flex-col h-full bg-background rounded-lg">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onCancel} className="md:hidden">
            <X className="w-5 h-5" />
          </Button>
          <Button variant="ghost" onClick={onCancel} className="hidden md:inline-flex">
            <X className="w-5 h-5 mr-2" />
            Close
          </Button>
          <h2 className="text-xl font-semibold m-0">
            {isNewTask ? 'New Task' : `Edit: ${originalName}`}
          </h2>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setShowYamlView(!showYamlView)}
          >
            {showYamlView ? 'Form View' : 'YAML View'}
          </Button>
          {!isNewTask && (
            <Button 
              variant="destructive"
              size="sm"
              onClick={() => setShowDeleteModal(true)}
              disabled={saving}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          )}
          <Button 
            size="sm"
            onClick={handleSave}
            disabled={saving || !name}
          >
            {saving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>

      {/* Validation errors */}
      {validationErrors.length > 0 && (
        <div className="mx-4 mt-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg">
          <div className="flex items-center gap-2 text-red-400 mb-2">
            <AlertCircle className="w-4 h-4" />
            <span className="font-medium">Please fix the following errors:</span>
          </div>
          <ul className="list-disc pl-5 text-sm text-red-300">
            {validationErrors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          {showYamlView ? (
            /* YAML View */
            <div className="space-y-4">
              <Label>YAML Configuration</Label>
              <textarea
                value={configText}
                onChange={(e) => setConfigText(e.target.value)}
                className="w-full h-[500px] font-mono text-sm p-4 bg-muted rounded-lg resize-none"
              />
            </div>
          ) : (
            /* Structured Form Fields */
            <div className="space-y-6">
              {/* Name Field */}
              <div className="space-y-2">
                <Label htmlFor="name">Task Name *</Label>
                <Input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => handleNameChange(e.target.value)}
                  placeholder="my-task-name"
                />
                <p className="text-sm text-muted-foreground">
                  Only lowercase letters, numbers, and hyphens allowed
                </p>
              </div>

              {/* Cron Field */}
              <div className="space-y-2">
                <Label htmlFor="cron">Cron Schedule *</Label>
                <Input
                  id="cron"
                  type="text"
                  value={cron}
                  onChange={(e) => {
                    setCron(e.target.value)
                    updateYamlFromForm()
                  }}
                  placeholder="0 * * * *"
                />
                <p className="text-sm text-muted-foreground">
                  Format: minute hour day month weekday (e.g., "0 * * * *" = every hour)
                </p>
                {!CRON_REGEX.test(cron) && cron && (
                  <p className="text-sm text-red-400">Invalid cron expression</p>
                )}
              </div>

              {/* Enabled Field */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id="enabled"
                  checked={enabled}
                  onCheckedChange={(checked) => {
                    setEnabled(checked === true)
                    updateYamlFromForm()
                  }}
                />
                <Label htmlFor="enabled" className="cursor-pointer">
                  Enabled
                </Label>
              </div>

              {/* ID Fields - Always separate user_id and channel_id */}
              <div className="space-y-4">
                <Label>Discord ID *</Label>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="userId">User ID</Label>
                    <Input
                      id="userId"
                      type="text"
                      value={userId}
                      onChange={(e) => {
                        setUserId(e.target.value)
                        updateYamlFromForm()
                      }}
                      placeholder="1234567890"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="channelId">Channel ID</Label>
                    <Input
                      id="channelId"
                      type="text"
                      value={channelId}
                      onChange={(e) => {
                        setChannelId(e.target.value)
                        updateYamlFromForm()
                      }}
                      placeholder="1234567890"
                    />
                  </div>
                </div>
              </div>

              {/* Model Field */}
              <div className="space-y-2">
                <Label htmlFor="model">Model</Label>
                <Select
                  value={model}
                  onValueChange={(value) => {
                    setModel(value)
                    updateYamlFromForm()
                  }}
                >
                  <SelectTrigger id="model">
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {AVAILABLE_MODELS.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))}
                    <SelectItem value="__custom__">Custom...</SelectItem>
                  </SelectContent>
                </Select>
                {model === '__custom__' && (
                  <Input
                    type="text"
                    value={model}
                    onChange={(e) => {
                      setModel(e.target.value)
                      updateYamlFromForm()
                    }}
                    placeholder="provider/model-name"
                    className="mt-2"
                  />
                )}
              </div>

              {/* Persona Field */}
              <div className="space-y-2">
                <Label htmlFor="persona">Persona</Label>
                <Select
                  value={persona}
                  onValueChange={(value) => {
                    setPersona(value === '__none__' ? '' : value)
                    updateYamlFromForm()
                  }}
                >
                  <SelectTrigger id="persona">
                    <SelectValue placeholder="Select persona (optional)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None</SelectItem>
                    {availablePersonas.map((p) => (
                      <SelectItem key={p} value={p}>
                        {p}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Tools Field */}
              <div className="space-y-2">
                <Label>Tools</Label>
                <div className="grid grid-cols-2 gap-2 p-3 border rounded-lg max-h-[200px] overflow-auto">
                  {availableTools.length > 0 ? (
                    availableTools.map((tool) => (
                      <div key={tool} className="flex items-center gap-2">
                        <Checkbox
                          id={`tool-${tool}`}
                          checked={tools.includes(tool)}
                          onCheckedChange={() => handleToolToggle(tool)}
                        />
                        <Label htmlFor={`tool-${tool}`} className="cursor-pointer text-sm">
                          {tool}
                        </Label>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-muted-foreground col-span-2">
                      No tools available
                    </p>
                  )}
                </div>
              </div>

              {/* Prompt Field */}
              <div className="space-y-2">
                <Label htmlFor="prompt">Prompt *</Label>
                <textarea
                  id="prompt"
                  value={prompt}
                  onChange={(e) => {
                    setPrompt(e.target.value)
                    updateYamlFromForm()
                  }}
                  placeholder="Your task prompt here..."
                  rows={6}
                  className="w-full p-3 border rounded-lg bg-background text-foreground resize-none"
                />
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteModal} onOpenChange={setShowDeleteModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Task</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{originalName}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteModal(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={saving}>
              {saving ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}