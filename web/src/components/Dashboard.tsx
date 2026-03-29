import { useState, useEffect, useRef } from 'react'
import { useAuth } from '../App'
import axios from 'axios'
import { RefreshCw, AlertTriangle } from 'lucide-react'
import { Button } from './ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog'
import LogViewer from './LogViewer'
import ConfigEditor from './ConfigEditor'
import ServerList from './ServerList'
import PersonaList from './PersonaList'
import PersonaEditor from './PersonaEditor'
import TaskList from './TaskList'
import TaskEditor from './TaskEditor'
import SkillsList from './SkillsList'
import Sidebar from './Sidebar'

const API_BASE = '/api'

interface BotStatus {
  status: string
  online: boolean
  uptime_seconds: number
  started_at: string
  server_count: number
  channel_count: number
  user_name?: string
  user_id?: number
  avatar_url?: string
  status_message?: string
}

const STORAGE_KEY = 'portal_active_tab'

function getInitialTab(): 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills' {
  const saved = localStorage.getItem(STORAGE_KEY)
  // Handle legacy 'status' key - redirect to 'dashboard'
  if (saved === 'dashboard' || saved === 'config' || saved === 'servers' || saved === 'personas' || saved === 'tasks' || saved === 'skills') {
    return saved as 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills'
  }
  if (saved === 'status' || saved === 'logs') {
    return 'dashboard'
  }
  return 'dashboard'
}

interface DashboardProps {
}

export default function Dashboard({ }: DashboardProps) {
  const { token, setToken } = useAuth()
  const [activeTab, setActiveTab] = useState<'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills'>(getInitialTab)
  const [status, setStatus] = useState<BotStatus | null>(null)
  const [loading, setLoading] = useState(true)
  // Local uptime counter that continues incrementing after initial data
  const [localUptime, setLocalUptime] = useState<number | null>(null)
  
  // Persona editor state
  const [editingPersona, setEditingPersona] = useState<string | null>(null)  // null = list view, string = editing persona name
  // Task editor state
  // null/undefined = list view, '' = new task, string = editing existing task
  const [editingTask, setEditingTask] = useState<string | null | undefined>(null)
  // Key to force TaskList re-render (reset expanded state) when tab is clicked
  const [tasksKey, setTasksKey] = useState(0)
  // Key to force TaskEditor re-render (refresh enabled checkbox after toggle)
  const [taskEditorKey, setTaskEditorKey] = useState(0)
  // Track unsaved changes in TaskEditor (Task 4.2.2)
  const [taskHasUnsavedChanges, setTaskHasUnsavedChanges] = useState(false)
  // Unsaved changes dialog state (Task 4.2.3)
  const [showUnsavedDialog, setShowUnsavedDialog] = useState(false)
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null)
  
  // Use ref to always get current token value
  const tokenRef = useRef(token)
  tokenRef.current = token

  const handleTabChange = (tab: 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills') => {
    // Reset to default view when clicking a tab, even if already on that tab
    // This ensures clicking a tab always goes to the list/default view (not previous state)
    // Reset persona editor to list view
    setEditingPersona(null)
    // Reset task editor to list view
    setEditingTask(undefined)
    // Reset TaskList expanded state by changing key
    if (tab === 'tasks') {
      setTasksKey(k => k + 1)
    }
    setActiveTab(tab)
    localStorage.setItem(STORAGE_KEY, tab)
  }

  // Handle request to create new task with unsaved changes check (Task 4.2.3)
  const handleRequestCreateTask = () => {
    if (taskHasUnsavedChanges) {
      // Show confirmation dialog
      setPendingAction(() => () => {
        setEditingTask('')
        setTaskHasUnsavedChanges(false)
      })
      setShowUnsavedDialog(true)
    } else {
      setEditingTask('')
    }
  }

  // Handle unsaved dialog actions (Task 4.2.3)
  const handleUnsavedDiscard = () => {
    setShowUnsavedDialog(false)
    setTaskHasUnsavedChanges(false)
    if (pendingAction) {
      pendingAction()
      setPendingAction(null)
    }
  }

  const handleUnsavedCancel = () => {
    setShowUnsavedDialog(false)
    setPendingAction(null)
  }

  const handleUnsavedSave = () => {
    // Close dialog - user should click Save in TaskEditor
    // The pending action remains queued, user can proceed after saving
    setShowUnsavedDialog(false)
    // TODO: Could implement auto-save by calling TaskEditor's save method
    // For now, user manually saves in TaskEditor, then can switch tasks
  }

  useEffect(() => {
    if (token) {
      fetchStatus()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  // Initialize local uptime from API and set up counter
  useEffect(() => {
    if (status?.uptime_seconds !== undefined) {
      setLocalUptime(status.uptime_seconds)
    }
  }, [status?.uptime_seconds])

  // Increment local uptime every minute
  useEffect(() => {
    if (localUptime === null) return
    
    const interval = setInterval(() => {
      setLocalUptime(prev => (prev ?? 0) + 60)
    }, 60000)
    
    return () => clearInterval(interval)
  }, [localUptime])

  // Periodic API refresh every 5 minutes to stay in sync
  useEffect(() => {
    if (!token) return
    
    const refreshInterval = setInterval(() => {
      fetchStatus()
    }, 300000) // 5 minutes
    
    return () => clearInterval(refreshInterval)
  }, [token])

  const fetchStatus = async () => {
    const currentToken = tokenRef.current
    if (!currentToken) {
      setLoading(false)
      return
    }
    try {
      const res = await axios.get(`${API_BASE}/status`, {
        headers: { Authorization: `Bearer ${currentToken}` }
      })
      setStatus(res.data)
      setLoading(false)
    } catch (err: any) {
      console.error('[Dashboard] Failed to fetch status:', err.response?.status, err.response?.data)
      setLoading(false)
    }
  }

  const handleLogout = () => {
    setToken(null)
  }

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  if (loading) {
    return <div>Loading...</div>
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar 
        activeTab={activeTab} 
        onTabChange={handleTabChange} 
        onLogout={handleLogout} 
      />

      <main className="flex-1 ml-[200px] p-6 max-w-[calc(100%-200px)] flex flex-col h-screen overflow-hidden box-border">
        {/* Dashboard: Show status + logs as widgets - combined into one view */}
        {activeTab === 'dashboard' && token && (
          <div className="flex flex-col gap-4 flex-1 min-h-0 overflow-hidden">
            {/* Status + Quick Actions combined card */}
            <div className="bg-card rounded-lg p-4 flex-shrink-0 flex gap-6 items-center">
              {/* Avatar on the left with status indicator */}
              <div className="relative flex-shrink-0">
                {status?.avatar_url ? (
                  <img 
                    src={status.avatar_url} 
                    alt={`${status.user_name} avatar`}
                    className="w-[60px] h-[60px] rounded-full"
                  />
                ) : (
                  <div className="w-[60px] h-[60px] rounded-full bg-muted flex items-center justify-center text-2xl">
                    🤖
                  </div>
                )}
                {/* Status indicator dot at bottom-right of avatar */}
                <span 
                  className="absolute bottom-0.5 right-0.5 w-3 h-3 rounded-full border-2 border-background"
                  style={{ backgroundColor: status?.online ? '#22c55e' : '#6b7280' }}
                  title={status?.online ? 'Online' : 'Offline'}
                />
              </div>
              
              {/* Info in the middle */}
              <div className="flex-1 flex gap-6 flex-wrap items-center">
                {status?.status_message && (
                  <span><strong>Status:</strong> {status.status_message}</span>
                )}
                <span><strong>Uptime:</strong> {localUptime !== null ? formatUptime(localUptime) : 'N/A'}</span>
                <span><strong>Guilds:</strong> {status?.server_count ?? 0}</span>
                <span><strong>Channels:</strong> {status?.channel_count ?? 0}</span>
              </div>

              {/* Quick Actions on the right */}
              <Button variant="outline" onClick={fetchStatus} className="flex-shrink-0">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
            
            {/* Real-time Logs Widget - fills remaining space */}
            <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
              <LogViewer token={token} />
            </div>
          </div>
        )}

        {activeTab === 'config' && token && (
          <div className="flex-1 min-h-0 overflow-auto">
            <ConfigEditor token={token} />
          </div>
        )}
        {activeTab === 'servers' && token && (
          <div className="flex-1 min-h-0 overflow-auto">
            <ServerList token={token} />
          </div>
        )}
        {activeTab === 'personas' && token && (
          editingPersona !== null ? (
            <div className="flex-1 min-h-0 overflow-auto">
              <PersonaEditor
                token={token}
                personaName={editingPersona || undefined}
                onSave={() => setEditingPersona(null)}
                onSaveWithName={(name) => setEditingPersona(name)}
                onCancel={() => setEditingPersona(null)}
              />
            </div>
          ) : (
            <div className="flex-1 min-h-0 overflow-auto">
              <PersonaList
                token={token}
                onSelectPersona={(name) => setEditingPersona(name)}
                onCreateNew={() => setEditingPersona('')}
              />
            </div>
          )
        )}

        {activeTab === 'tasks' && token && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col md:flex-row gap-4">
            {/* Left panel: TaskList - 50% on desktop, 40% on tablet, full on mobile */}
            {/* On mobile: hide when TaskEditor is active (editingTask is set) */}
            <div className={`w-full md:w-1/2 lg:w-[40%] min-h-[300px] md:min-h-0 overflow-hidden flex flex-col ${editingTask ? 'hidden md:flex' : 'flex'}`}>
              <TaskList 
                key={tasksKey}
                token={token} 
                editingTask={editingTask}
                onRequestCreate={handleRequestCreateTask}
                // Section 13: Task Dashboard Improvements
                onExpand={(name) => {
                  // Empty string means close editor (collapse card)
                  if (name === '') {
                    setEditingTask(undefined)
                  } else {
                    setEditingTask(name)
                  }
                }}
                onRequestExpand={(name) => {
                  if (taskHasUnsavedChanges) {
                    setPendingAction(() => () => {
                      setEditingTask(name)
                      setTaskHasUnsavedChanges(false)
                    })
                    setShowUnsavedDialog(true)
                  } else {
                    setEditingTask(name)
                  }
                }}
                // Refresh TaskEditor when task is modified (toggle, delete, etc.)
                onTaskChange={() => setTaskEditorKey(k => k + 1)}
              />
            </div>
            {/* Right panel: TaskEditor - 50% on desktop, 60% on tablet, full on mobile */}
            {/* On mobile: show full width when active, hidden when no task selected */}
            <div className={`w-full md:w-1/2 lg:w-[60%] min-h-[400px] md:min-h-0 overflow-hidden flex flex-col ${editingTask ? 'flex' : 'hidden md:flex'}`}>
              <TaskEditor
                key={taskEditorKey}
                token={token}
                taskName={editingTask === '' ? undefined : editingTask}
                onSave={() => {
                  setEditingTask(undefined)
                  // Refresh task list to show updated status
                  setTasksKey(k => k + 1)
                }}
                onSaveWithName={(name) => {
                  setEditingTask(name || undefined)
                  // Refresh task list to show new task
                  setTasksKey(k => k + 1)
                }}
                onCancel={() => setEditingTask(undefined)}
                onDirtyChange={(isDirty) => setTaskHasUnsavedChanges(isDirty)}
              />
            </div>
          </div>
        )}

        {activeTab === 'skills' && token && (
          <div className="flex-1 min-h-0 overflow-auto">
            <SkillsList token={token} />
          </div>
        )}

        {/* Unsaved Changes Confirmation Dialog (Task 4.2.3, 4.3) */}
        <Dialog open={showUnsavedDialog} onOpenChange={setShowUnsavedDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-500" />
                Unsaved Changes
              </DialogTitle>
              <DialogDescription>
                You have unsaved changes in the current task. What would you like to do?
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="flex flex-col sm:flex-row gap-2">
              <Button variant="outline" onClick={handleUnsavedCancel} className="flex-1">
                Cancel
              </Button>
              <Button variant="default" onClick={handleUnsavedSave} className="flex-1">
                Save
              </Button>
              <Button variant="destructive" onClick={handleUnsavedDiscard} className="flex-1">
                Discard
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </main>
    </div>
  )
}