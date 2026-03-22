import { useState, useEffect } from 'react'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import FormModal from './FormModal'
import {
  Plus,
  Pencil,
  Trash2,
  Eye,
  EyeOff,
} from 'lucide-react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Tabs, TabsList, TabsTrigger } from './ui/tabs'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'


const API_BASE = '/api'

// Config sections for sidebar
const CONFIG_SECTIONS = [
  { id: 'discord', label: 'Discord', icon: '💬', fields: ['discord.status_message', 'discord.bot_token', 'discord.client_id', 'discord.permissions'] },
  { id: 'behavior', label: 'Behavior', icon: '⚙️', fields: ['behavior.max_text', 'behavior.max_images', 'behavior.max_messages', 'behavior.use_plain_responses', 'behavior.show_embed_color', 'behavior.allow_dms'] },
  { id: 'llm', label: 'LLM', icon: '🤖', fields: ['llm.providers', 'llm.models', 'llm.fallback_models', 'llm.persona', 'llm.system_prompt'] },
  { id: 'voice', label: 'Voice', icon: '🎤', fields: ['voice.region', 'voice.default_voice', 'voice.key'] },
  { id: 'portal', label: 'Web Portal', icon: '🌐', fields: ['portal.enabled', 'portal.port', 'portal.cors_origins', 'portal.docs_enabled', 'portal.require_discord_admin', 'portal.logs.retention_days', 'portal.logs.levels'] },
]

// LLM tabs
const LLM_TABS = ['Providers', 'Models']

// Available tools for model configuration
const AVAILABLE_TOOLS = ['web_search', 'get_market_prices', 'get_weather', 'google_tools', 'visuals_core']

// Persona options
const PERSONA_OPTIONS = ['bao', 'default']

interface ConfigField {
  key: string
  value: any
  editable: boolean
  type: string
  options?: string[]  // For checkbox-group type fields
}

interface ConfigSection {
  id: string
  label: string
  icon: string
  fields: ConfigField[]
}

interface Provider {
  name: string
  base_url: string
  api_key?: string
}

interface Model {
  name: string
  persona: string
  supports_tools: boolean
  tools: string[]
  think: boolean
}

interface ConfigEditorProps {
  token: string
}

export default function ConfigEditor({ token }: ConfigEditorProps) {
  const [sections, setSections] = useState<ConfigSection[]>([])
  const [activeSection, setActiveSection] = useState<string>('discord')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  
  // LLM section state
  const [llmTab, setLlmTab] = useState<string>('Providers')
  const [providers, setProviders] = useState<Provider[]>([])
  const [models, setModels] = useState<Model[]>([])
  
  // Modal state
  const [showProviderModal, setShowProviderModal] = useState(false)
  const [showModelModal, setShowModelModal] = useState(false)
  const [editingProvider, setEditingProvider] = useState<Provider | null>(null)
  const [editingModel, setEditingModel] = useState<Model | null>(null)
  
  // JSON mode state (Task 9.4)
  const [editMode, setEditMode] = useState<'form' | 'json'>('form')
  const [jsonContent, setJsonContent] = useState('')
  const [jsonError, setJsonError] = useState<string | null>(null)
  const [fullConfig, setFullConfig] = useState<Record<string, any>>({})
  
  // API key and bot_token visibility toggle
  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({})
  const [showBotToken, setShowBotToken] = useState(false)
  const [showVoiceKey, setShowVoiceKey] = useState(false)
  const [showProviderApiKey, setShowProviderApiKey] = useState(false)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Tailwind class for input fields (replaces inputStyle)
  const inputClass = "h-10 bg-background border border-input rounded-md px-3 text-sm text-foreground flex-1 outline-none focus:border-primary transition-colors"

  useEffect(() => {
    fetchConfig()
  }, [])

  // Update JSON content when section changes - display full config
  useEffect(() => {
    if (editMode === 'json') {
      // Display the entire config.yaml content in JSON mode
      setJsonContent(JSON.stringify(fullConfig, null, 2))
      setJsonError(null)
    }
  }, [activeSection, editMode, fullConfig])

  const fetchConfig = async () => {
    try {
      const res = await axios.get(`${API_BASE}/config`, authHeaders)
      const apiData = res.data
      
      // Collect all fields from both editable and read-only
      const allFields: ConfigField[] = []
      
      // Use unredacted discord_config for sensitive fields (bot_token)
      const discordConfig = apiData.discord_config || {}
      
      // Portal config is in a separate field (apiData.portal)
      const portalConfig = apiData.portal || {}
      // Voice config is in a separate field (apiData.voice)
      const voiceConfig = apiData.voice || {}
      
      for (const key of apiData.editable_fields || []) {
        let value: any = undefined
        
        // Handle portal nested fields (from separate apiData.portal)
        if (key.startsWith('portal.')) {
          const portalKey = key.replace('portal.', '')
          const portalParts = portalKey.split('.')
          value = portalParts.reduce((acc: any, part: string) => acc && acc[part], portalConfig)
        }
        // Handle voice nested fields (from separate apiData.voice)
        else if (key.startsWith('voice.')) {
          const voiceKey = key.replace('voice.', '')
          value = voiceConfig[voiceKey]
        }
        // Regular config fields
        else {
          value = getNestedValue(apiData.config, key)
        }
        
        // Use unredacted bot_token from discord_config
        if (key === 'discord.bot_token' && discordConfig.bot_token !== undefined) {
          value = discordConfig.bot_token
        }
        
        if (value !== undefined) {
          // Determine field type based on key
          let fieldType: string = typeof value
          
          // Use checkbox-group for portal.logs.levels (array of strings)
          if (key === 'portal.logs.levels' && Array.isArray(value)) {
            fieldType = 'checkbox-group'
          }
          
          // Use password for voice.key (sensitive)
          if (key === 'voice.key') {
            fieldType = 'password'
          }
          
          allFields.push({ key, value, editable: true, type: fieldType, options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'] as string[] })
        }
      }
      
      for (const key of apiData.read_only_fields || []) {
        // Handle voice in read_only_fields (maps to config.voice)
        if (key === 'voice') {
          const value = apiData.config?.voice
          if (value !== undefined) {
            allFields.push({ key, value, editable: false, type: typeof value })
          }
        } else {
          const value = getNestedValue(apiData.config, key)
          if (value !== undefined) {
            allFields.push({ key, value, editable: false, type: typeof value })
          }
        }
      }
      
      // Group fields by section
      console.log('allFields:', allFields)
      const groupedSections = CONFIG_SECTIONS.map(section => {
        const sectionFields = allFields.filter(field => {
          if (section.fields.includes(field.key)) {
            return true
          }
          return false
        })
        return { ...section, fields: sectionFields }
      }).filter(s => s.fields.length > 0)
      console.log('groupedSections:', groupedSections.map(s => s.id))
      
      setSections(groupedSections)
      if (groupedSections.length > 0) {
        setActiveSection(groupedSections[0].id)
      }
      
      // Load structured LLM data (Task 9.2.7)
      if (apiData.llm_providers) {
        const providerList = Object.entries(apiData.llm_providers).map(([name, data]: [string, any]) => ({
          name,
          base_url: data.base_url || '',
          api_key: data.api_key || ''
        }))
        setProviders(providerList)
      }
      
      if (apiData.llm_models) {
        const modelList = Object.entries(apiData.llm_models).map(([name, data]: [string, any]) => ({
          name,
          persona: data.persona || 'default',
          supports_tools: data.supports_tools || false,
          tools: data.tools || [],
          think: data.think || false
        }))
        setModels(modelList)
      }
      
      // Store full config for JSON mode
      if (apiData.full_config) {
        setFullConfig(apiData.full_config)
      }
      
    } catch (err) {
      console.error('Failed to fetch config:', err)
    } finally {
      setLoading(false)
    }
  }

  // Helper to get nested value from config object
  const getNestedValue = (obj: any, path: string): any => {
    if (!obj) return undefined
    return path.split('.').reduce((acc, part) => acc && acc[part], obj)
  }

  // Task 9.4.3: Parse JSON back to form data
  const jsonToForm = (json: string): boolean => {
    try {
      const parsed = JSON.parse(json)
      
      // Update sections with parsed values
      setSections(prev => prev.map(section => ({
        ...section,
        fields: section.fields.map(field => {
          const value = getNestedValue(parsed, field.key)
          if (value !== undefined) {
            return { ...field, value }
          }
          return field
        })
      })))
      
      setJsonError(null)
      return true
    } catch (e: any) {
      setJsonError(e.message)
      return false
    }
  }

  const handleSave = async () => {
    setSaving(true)
    
    try {
      const updates = getAllFields()
        .filter(f => f.editable)
        .map(f => ({ key: f.key, value: f.value }))
      
      await axios.put(`${API_BASE}/config`, { fields: updates }, authHeaders)
      toast.success('Configuration saved successfully!')
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save config')
    } finally {
      setSaving(false)
    }
  }

  const handleApply = async () => {
    setSaving(true)
    
    try {
      await axios.post(`${API_BASE}/refresh`, {}, authHeaders)
      toast.success('Configuration applied (in-memory reload)!')
      fetchConfig()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to apply config')
    } finally {
      setSaving(false)
    }
  }

  const handleSaveAndApply = async () => {
    setSaving(true)
    
    try {
      const updates = getAllFields()
        .filter(f => f.editable)
        .map(f => ({ key: f.key, value: f.value }))
      
      await axios.put(`${API_BASE}/config`, { fields: updates }, authHeaders)
      await axios.post(`${API_BASE}/refresh`, {}, authHeaders)
      
      toast.success('Configuration saved and applied!')
      fetchConfig()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save and apply config')
    } finally {
      setSaving(false)
    }
  }

  const updateField = (sectionId: string, key: string, value: any) => {
    setSections(prev => prev.map(section => 
      section.id === sectionId 
        ? { ...section, fields: section.fields.map(f => 
            f.key === key ? { ...f, value } : f 
          )}
        : section
    ))
  }

  const getAllFields = (): ConfigField[] => {
    return sections.flatMap(s => s.fields)
  }

  // List-type fields that should be rendered as comma-separated inputs
  const LIST_FIELDS = ['llm.fallback_models', 'portal.logs.levels', 'portal.cors_origins']
  
  const isListField = (key: string): boolean => {
    return LIST_FIELDS.includes(key)
  }

  // Get friendly display name for a field key
  const getFieldLabel = (key: string): string => {
    const labelMap: Record<string, string> = {
      // Discord section
      'discord.status_message': 'Status Message',
      'discord.bot_token': 'Bot Token',
      'discord.client_id': 'Client ID',
      'discord.permissions': 'Permissions',
      // Behavior section
      'behavior.max_text': 'Max Text Characters',
      'behavior.max_images': 'Max Images',
      'behavior.max_messages': 'Max Messages',
      'behavior.use_plain_responses': 'Plain Responses',
      'behavior.show_embed_color': 'Show Embed Color',
      'behavior.allow_dms': 'Allow DMs',
      // Voice section
      'voice.key': 'API Key',
      'voice.default_voice': 'Default Voice',
      'voice.region': 'Region',
      // Web Portal section
      'portal.enabled': 'Enabled',
      'portal.port': 'Port',
      'portal.cors_origins': 'CORS Origins',
      'portal.docs_enabled': 'Docs Enabled',
      'portal.require_discord_admin': 'Require Discord Admin',
      'portal.logs.retention_days': 'Log Retention (days)',
      'portal.logs.levels': 'Log Levels',
    }
    return labelMap[key] || key.split('.').pop() || key
  }
  
  const renderInput = (sectionId: string, field: ConfigField) => {
    if (!field.editable) {
      // For display of list values, join with comma
      if (Array.isArray(field.value)) {
        return <span>{field.value.join(', ')}</span>
      }
      return <span>{String(field.value)}</span>
    }

    // Handle list-type fields (render as comma-separated string)
    if (isListField(field.key) && Array.isArray(field.value)) {
      return (
        <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          <input
            type="text"
            value={field.value.join(', ')}
            onChange={(e) => {
              // Convert comma-separated string to array
              const arr = e.target.value.split(',').map(s => s.trim()).filter(s => s)
              updateField(sectionId, field.key, arr)
            }}
            className={inputClass}
            placeholder="model1, model2, model3"
          />
        </div>
      )
    }

    switch (field.type as string) {
      case 'boolean':
        return (
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={Boolean(field.value)}
              onChange={(e) => updateField(sectionId, field.key, e.target.checked)}
            />
            <span>{String(field.value)}</span>
          </label>
        )
      case 'checkbox-group':
        // Render checkbox group for multi-select fields like portal.logs.levels
        const options = field.options || []
        const currentValues = Array.isArray(field.value) ? field.value : []
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {options.map((option: string) => (
              <label 
                key={option} 
                style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}
              >
                <input
                  type="checkbox"
                  checked={currentValues.includes(option)}
                  onChange={(e) => {
                    let newValues: string[]
                    if (e.target.checked) {
                      newValues = [...currentValues, option]
                    } else {
                      newValues = currentValues.filter((v: string) => v !== option)
                    }
                    updateField(sectionId, field.key, newValues)
                  }}
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        )
      case 'number':
        return (
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Input
              type="number"
              value={field.value}
              onChange={(e) => updateField(sectionId, field.key, Number(e.target.value))}
              className={inputClass}
            />
          </div>
        )
      case 'password':
        // Special handling for voice.key - show/hide toggle
        if (field.key === 'voice.key') {
          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
              <Input
                type={showVoiceKey ? 'text' : 'password'}
                value={field.value}
                onChange={(e) => updateField(sectionId, field.key, e.target.value)}
                className={inputClass}
              />
              <Button 
                variant="ghost" 
                size="icon"
                onClick={() => setShowVoiceKey(!showVoiceKey)}
                title={showVoiceKey ? 'Hide' : 'Show'}
              >
                {showVoiceKey ? (
                  <EyeOff className="w-5 h-5" />
                ) : (
                  <Eye className="w-5 h-5" />
                )}
              </Button>
            </div>
          )
        }
        // Default password input for other password fields
        return (
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Input
              type="password"
              value={field.value}
              onChange={(e) => updateField(sectionId, field.key, e.target.value)}
              className={inputClass}
            />
          </div>
        )
      default:
        // Special handling for discord.bot_token - show/hide toggle
        if (field.key === 'discord.bot_token') {
          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
              <Input
                type={showBotToken ? 'text' : 'password'}
                value={field.value}
                onChange={(e) => updateField(sectionId, field.key, e.target.value)}
                className={inputClass}
              />
              <Button 
                variant="ghost" 
                size="icon"
                onClick={() => setShowBotToken(!showBotToken)}
                title={showBotToken ? 'Hide' : 'Show'}
              >
                {showBotToken ? (
                  <EyeOff className="w-5 h-5" />
                ) : (
                  <Eye className="w-5 h-5" />
                )}
              </Button>
            </div>
          )
        }
        // Consistent text input with flex wrapper
        return (
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Input
              type="text"
              value={field.value}
              onChange={(e) => updateField(sectionId, field.key, e.target.value)}
              className={inputClass}
            />
          </div>
        )
    }
  }

  // Provider CRUD handlers
  const handleAddProvider = () => {
    setEditingProvider({ name: '', base_url: '', api_key: '' })
    setShowProviderModal(true)
  }

  const handleEditProvider = (provider: Provider) => {
    setEditingProvider({ ...provider })
    setShowProviderModal(true)
  }

  const handleDeleteProvider = async (name: string) => {
    if (!confirm(`Delete provider "${name}"?`)) return
    
    try {
      await axios.put(`${API_BASE}/config/providers`, {
        action: 'delete',
        name
      }, authHeaders)
      toast.success(`Provider "${name}" deleted`)
      fetchConfig()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to delete provider')
    }
  }

  const handleSaveProvider = async () => {
    if (!editingProvider) return
    
    try {
      const action = providers.find(p => p.name === editingProvider.name) ? 'update' : 'add'
      await axios.put(`${API_BASE}/config/providers`, {
        action,
        name: editingProvider.name,
        data: {
          base_url: editingProvider.base_url,
          api_key: editingProvider.api_key
        }
      }, authHeaders)
      
      toast.success(`Provider "${editingProvider.name}" ${action === 'add' ? 'added' : 'updated'}`)
      setShowProviderModal(false)
      fetchConfig()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save provider')
    }
  }

  // Model CRUD handlers
  const handleAddModel = () => {
    setEditingModel({ name: '', persona: 'default', supports_tools: false, tools: [], think: false })
    setShowModelModal(true)
  }

  const handleEditModel = (model: Model) => {
    setEditingModel({ ...model })
    setShowModelModal(true)
  }

  const handleDeleteModel = async (name: string) => {
    if (!confirm(`Delete model "${name}"?`)) return
    
    try {
      await axios.put(`${API_BASE}/config/models`, {
        action: 'delete',
        model_name: name
      }, authHeaders)
      toast.success(`Model "${name}" deleted`)
      fetchConfig()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to delete model')
    }
  }

  const handleSaveModel = async () => {
    if (!editingModel) return
    
    try {
      const existingModel = models.find(m => m.name === editingModel.name)
      const originalName = editingModel.name
      
      // If name changed and this is an existing model, we need to handle it
      if (existingModel && existingModel.name !== originalName) {
        // Delete old model first, then add new one
        await axios.put(`${API_BASE}/config/models`, {
          action: 'delete',
          model_name: existingModel.name
        }, authHeaders)
      }
      
      const action = existingModel && existingModel.name === originalName ? 'update' : 'add'
      await axios.put(`${API_BASE}/config/models`, {
        action,
        model_name: editingModel.name,
        data: {
          persona: editingModel.persona,
          supports_tools: editingModel.supports_tools,
          tools: editingModel.tools,
          think: editingModel.think
        }
      }, authHeaders)
      
      toast.success(`Model "${editingModel.name}" ${action === 'add' ? 'added' : 'updated'}`)
      setShowModelModal(false)
      fetchConfig()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save model')
    }
  }

  const handleToggleEditMode = () => {
    if (editMode === 'form') {
      // Form → JSON: display the entire config.yaml content
      setJsonContent(JSON.stringify(fullConfig, null, 2))
      setEditMode('json')
    } else {
      // JSON → Form
      if (jsonToForm(jsonContent)) {
        setEditMode('form')
      }
    }
  }

  const renderProviderModal = () => (
    <FormModal
      isOpen={showProviderModal}
      onClose={() => setShowProviderModal(false)}
      title={editingProvider?.name ? 'Edit Provider' : 'Add Provider'}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#fff' }}>Name</label>
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Input
              type="text"
              value={editingProvider?.name || ''}
              onChange={(e) => setEditingProvider(prev => prev ? { ...prev, name: e.target.value } : null)}
              disabled={!!providers.find(p => p.name === editingProvider?.name)}
              className={inputClass}
            />
          </div>
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>Base URL</label>
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Input
              type="text"
              value={editingProvider?.base_url || ''}
              onChange={(e) => setEditingProvider(prev => prev ? { ...prev, base_url: e.target.value } : null)}
              className={inputClass}
            />
          </div>
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>API Key</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
            <Input
              type={showProviderApiKey ? 'text' : 'password'}
              value={editingProvider?.api_key || ''}
              onChange={(e) => setEditingProvider(prev => prev ? { ...prev, api_key: e.target.value } : null)}
              className={inputClass}
            />
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => setShowProviderApiKey(!showProviderApiKey)}
              title={showProviderApiKey ? 'Hide' : 'Show'}
            >
              {showProviderApiKey ? (
                <EyeOff style={{ width: '20px', height: '20px' }} />
              ) : (
                <Eye style={{ width: '20px', height: '20px' }} />
              )}
            </Button>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <Button 
            variant="secondary"
            onClick={() => setShowProviderModal(false)}
          >
            Cancel
          </Button>
          <Button 
            variant="secondary"
            onClick={handleSaveProvider}
          >
            Save
          </Button>
        </div>
      </div>
    </FormModal>
  )

  const renderModelModal = () => (
    <FormModal
      isOpen={showModelModal}
      onClose={() => setShowModelModal(false)}
      title={editingModel?.name ? 'Edit Model' : 'Add Model'}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#fff' }}>Model Name</label>
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Input
              type="text"
              value={editingModel?.name || ''}
              onChange={(e) => setEditingModel(prev => prev ? { ...prev, name: e.target.value } : null)}
              className={inputClass}
            />
          </div>
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>Persona</label>
          <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <Select
              value={editingModel?.persona || 'default'}
              onValueChange={(value) => setEditingModel(prev => prev ? { ...prev, persona: value } : null)}
            >
              <SelectTrigger className={inputClass}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PERSONA_OPTIONS.map(p => (
                  <SelectItem key={p} value={p}>{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="checkbox"
              checked={editingModel?.supports_tools || false}
              onChange={(e) => setEditingModel(prev => prev ? { ...prev, supports_tools: e.target.checked } : null)}
            />
            Supports Tools
          </label>
        </div>
        <div>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="checkbox"
              checked={editingModel?.think || false}
              onChange={(e) => setEditingModel(prev => prev ? { ...prev, think: e.target.checked } : null)}
            />
            Think
          </label>
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '0.5rem' }}>Tools</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {AVAILABLE_TOOLS.map(tool => (
              <label key={tool} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={editingModel?.tools?.includes(tool) || false}
                  onChange={(e) => {
                    const currentTools = editingModel?.tools || []
                    const newTools = e.target.checked
                      ? [...currentTools, tool]
                      : currentTools.filter(t => t !== tool)
                    setEditingModel(prev => prev ? { ...prev, tools: newTools } : null)
                  }}
                />
                {tool}
              </label>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <Button 
            variant="secondary"
            onClick={() => setShowModelModal(false)}
          >
            Cancel
          </Button>
          <Button 
            variant="secondary"
            onClick={handleSaveModel}
          >
            Save
          </Button>
        </div>
      </div>
    </FormModal>
  )

  const renderLLMTabContent = () => {
    if (llmTab === 'Providers') {
      return (
        <div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
            <button 
              onClick={handleAddProvider}
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '6px',
                backgroundColor: '#4b5563',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '8px 16px',
                cursor: 'pointer',
                fontSize: '0.9em'
              }}
            >
              <Plus className="h-4 w-4" />
              Add Provider
            </button>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Base URL</TableHead>
                <TableHead>API Key</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {providers.map(provider => (
                <TableRow key={provider.name}>
                  <TableCell>{provider.name}</TableCell>
                  <TableCell className="text-muted-foreground text-sm">{provider.base_url}</TableCell>
                  <TableCell>
                    {provider.api_key ? (
                      <div className="flex items-center gap-2">
                        <code className="text-sm">{showApiKeys[provider.name] ? provider.api_key : '••••••••'}</code>
                        <Button 
                          variant="ghost" 
                          size="icon"
                          onClick={() => setShowApiKeys(prev => ({ ...prev, [provider.name]: !prev[provider.name] }))}
                          title={showApiKeys[provider.name] ? 'Hide' : 'Show'}
                        >
                          {showApiKeys[provider.name] ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </Button>
                      </div>
                    ) : '(empty)'}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => handleEditProvider(provider)} 
                      title="Edit"
                    >
                      <Pencil className="w-4 h-4" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => handleDeleteProvider(provider.name)}
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {providers.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-muted-foreground">
                    No providers configured. Click "Add Provider" to create one.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      )
    }
    
    if (llmTab === 'Models') {
      return (
        <div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
            <button 
              onClick={handleAddModel}
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '6px',
                backgroundColor: '#4b5563',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '8px 16px',
                cursor: 'pointer',
                fontSize: '0.9em'
              }}
            >
              <Plus className="h-4 w-4" />
              Add Model
            </button>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model Name</TableHead>
                <TableHead>Persona</TableHead>
                <TableHead className="text-center">Tools</TableHead>
                <TableHead className="text-center">Supports Tools</TableHead>
                <TableHead className="text-center">Think</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map(model => (
                <TableRow key={model.name}>
                  <TableCell>{model.name}</TableCell>
                  <TableCell>{model.persona}</TableCell>
                  <TableCell className="text-sm">
                    {model.tools?.join(', ') || '-'}
                  </TableCell>
                  <TableCell className="text-center">
                    {model.supports_tools ? '✓' : '-'}
                  </TableCell>
                  <TableCell className="text-center">
                    {model.think ? '✓' : '-'}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => handleEditModel(model)} 
                      title="Edit"
                    >
                      <Pencil className="w-4 h-4" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="icon"
                      onClick={() => handleDeleteModel(model.name)}
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {models.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground">
                    No models configured. Click "Add Model" to create one.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      )
    }
    
    // Persona tab - show persona-related config fields
    const personaFields = sections.find(s => s.id === 'llm')?.fields.filter(f => 
      f.key === 'llm.persona' || f.key === 'llm.system_prompt' || f.key === 'llm.fallback_models'
    ) || []
    
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {personaFields.map(field => (
          <div key={field.key}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#888' }}>
              {getFieldLabel(field.key)}
            </label>
            {renderInput('llm', field)}
          </div>
        ))}
        {personaFields.length === 0 && (
          <p style={{ color: '#666' }}>No persona configuration available.</p>
        )}
      </div>
    )
  }

  if (loading) {
    return <div>Loading...</div>
  }

  const activeContent = sections.find(s => s.id === activeSection)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header with title and edit mode toggle */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '1rem',
        padding: '0 0.5rem'
      }}>
        <h2 style={{ margin: 0 }}>Configuration</h2>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button 
            onClick={handleToggleEditMode}
            style={{ 
              backgroundColor: editMode === 'json' ? '#646cff' : '#2a2a2a',
              color: '#fff',
              border: '1px solid #444',
              borderRadius: '6px',
              padding: '6px 12px',
              cursor: 'pointer',
              fontSize: '0.85rem',
              transition: 'background-color 0.2s'
            }}
          >
            {editMode === 'form' ? 'JSON' : 'Form'}
          </button>
        </div>
      </div>

      {/* Main content area: Sidebar + Content */}
      <div style={{ display: 'flex', flex: 1, gap: '1rem', overflow: 'hidden' }}>
        {/* Sidebar */}
        <div style={{ 
          width: '200px', 
          flexShrink: 0,
          backgroundColor: '#1a1a1a', 
          borderRadius: '8px',
          padding: '0.5rem',
          overflow: 'auto'
        }}>
          {sections.map(section => (
            <div
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              style={{
                padding: '12px',
                cursor: 'pointer',
                borderRadius: '8px',
                marginBottom: '0.25rem',
                backgroundColor: activeSection === section.id ? '#2a2a2a' : 'transparent',
                border: activeSection === section.id ? '1px solid #444' : '1px solid transparent',
              }}
            >
              <span style={{ marginRight: '0.5rem' }}>{section.icon}</span>
              {section.label}
            </div>
          ))}
        </div>

        {/* Content panel */}
        <div style={{ 
          flex: 1, 
          backgroundColor: '#1a1a1a', 
          borderRadius: '8px',
          padding: '1rem',
          overflow: 'auto'
        }}>
          {editMode === 'json' ? (
            /* JSON Mode (Task 9.4) */
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              {jsonError && (
                <div style={{ 
                  color: '#ff6b6b', 
                  marginBottom: '0.5rem',
                  padding: '0.5rem',
                  backgroundColor: 'rgba(255, 107, 107, 0.1)',
                  borderRadius: '4px'
                }}>
                  {jsonError}
                </div>
              )}
              <textarea
                value={jsonContent}
                onChange={(e) => {
                  setJsonContent(e.target.value)
                  setJsonError(null)
                }}
                style={{
                  flex: 1,
                  width: '100%',
                  resize: 'none',
                  padding: '1rem',
                  backgroundColor: '#1a1a1a',
                  border: jsonError ? '1px solid #ff6b6b' : '1px solid #444',
                  borderRadius: '4px',
                  color: '#ddd',
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                  lineHeight: '1.5',
                }}
              />
            </div>
          ) : (
            /* Form Mode */
            <>
              {/* Section title */}
              {activeContent && (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
                    <span style={{ fontSize: '1.5rem', marginRight: '0.5rem' }}>{activeContent.icon}</span>
                    <h3 style={{ margin: 0 }}>{activeContent.label}</h3>
                  </div>
                  
                  {/* LLM section has tabs */}
                  {activeSection === 'llm' ? (
                    <Tabs value={llmTab} onValueChange={setLlmTab}>
                      <TabsList className="mb-4">
                        {LLM_TABS.map(tab => (
                          <TabsTrigger key={tab} value={tab}>{tab}</TabsTrigger>
                        ))}
                      </TabsList>
                      
                      {/* Tab content */}
                      {renderLLMTabContent()}
                    </Tabs>
                  ) : (
                    /* Regular fields */
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      {activeContent.fields.map(field => (
                        <div key={field.key}>
                          <label style={{ 
                            display: 'block', 
                            marginBottom: '0.5rem', 
                            color: '#888',
                            fontSize: '0.9em'
                          }}>
                            {getFieldLabel(field.key)}
                            {!field.editable && (
                              <span style={{ color: '#666', marginLeft: '0.5rem' }}>(read-only)</span>
                            )}
                          </label>
                          <div style={{ width: '100%' }}>
                            {renderInput(activeSection, field)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '12px', marginTop: '1rem', justifyContent: 'flex-end', flexShrink: 0 }}>
        <button 
          onClick={handleSave} 
          disabled={saving}
          style={{ 
            backgroundColor: '#4b5563', 
            color: '#fff',
            border: '1px solid #555',
            borderRadius: '6px',
            padding: '0 16px',
            height: '36px',
            minWidth: '80px',
            cursor: saving ? 'not-allowed' : 'pointer',
            opacity: saving ? 0.5 : 1,
            transition: 'opacity 0.2s, background-color 0.2s'
          }}
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button 
          onClick={handleApply} 
          disabled={saving}
          style={{ 
            backgroundColor: '#4b5563', 
            color: '#fff',
            border: '1px solid #555',
            borderRadius: '6px',
            padding: '0 16px',
            height: '36px',
            minWidth: '80px',
            cursor: saving ? 'not-allowed' : 'pointer',
            opacity: saving ? 0.5 : 1,
            transition: 'opacity 0.2s, background-color 0.2s'
          }}
        >
          {saving ? 'Applying...' : 'Apply'}
        </button>
        <button 
          onClick={handleSaveAndApply} 
          disabled={saving}
          style={{ 
            backgroundColor: '#646cff', 
            color: '#fff',
            border: '1px solid #747bff',
            borderRadius: '6px',
            padding: '0 16px',
            height: '36px',
            minWidth: '100px',
            cursor: saving ? 'not-allowed' : 'pointer',
            opacity: saving ? 0.5 : 1,
            transition: 'opacity 0.2s, background-color 0.2s'
          }}
        >
          {saving ? 'Saving...' : 'Save&Apply'}
        </button>
      </div>

      {/* Modals */}
      {renderProviderModal()}
      {renderModelModal()}
    </div>
  )
}