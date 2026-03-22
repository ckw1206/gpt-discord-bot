import { useState, useEffect } from 'react'
import axios from 'axios'
import { toast } from 'react-hot-toast'
import { 
  RefreshCw, 
  ChevronDown, 
  ChevronRight, 
  Pencil, 
  Plus, 
  Play, 
  Clock
} from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'

const API_BASE = '/api'

interface TaskInfo {
  name: string
  filename: string
  enabled: boolean | null
  schedule: string | null
  description: string | null
  status: string
}

interface TaskListProps {
  token: string
  onSelectTask?: (name: string) => void
  onCreateNew?: () => void
  // If provided, show editor inline instead of callbacks
  editingTask?: string | null
  onEditComplete?: () => void
  onRequestEdit?: (name: string) => void
  onRequestCreate?: () => void
}

export default function TaskList({ token, onSelectTask, onCreateNew, onRequestEdit, onRequestCreate }: TaskListProps) {
  const [tasks, setTasks] = useState<TaskInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedTask, setExpandedTask] = useState<string | null>(null)
  const [toggling, setToggling] = useState<string | null>(null)
  const [runningTask, setRunningTask] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchTasks()
  }, [])

  const fetchTasks = async () => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/tasks`, authHeaders)
      setTasks(res.data || [])
      setError(null)
    } catch (err: any) {
      console.error('Failed to fetch tasks:', err)
      setError(err.response?.data?.detail || 'Failed to load tasks')
    } finally {
      setLoading(false)
    }
  }

  const toggleTask = async (taskName: string, currentEnabled: boolean) => {
    try {
      setToggling(taskName)
      // Fetch full task config first
      const res = await axios.get(`${API_BASE}/tasks/${taskName}`, authHeaders)
      const config = res.data.config
      config.enabled = !currentEnabled
      
      // Update the task - wrap in { config: ... } for Pydantic model
      await axios.put(`${API_BASE}/tasks/${taskName}`, { config }, authHeaders)
      
      // Refresh the list
      await fetchTasks()
    } catch (err: any) {
      console.error('Failed to toggle task:', err)
      setError(err.response?.data?.detail || 'Failed to toggle task')
    } finally {
      setToggling(null)
    }
  }

  const runTask = async (taskName: string) => {
    let jobId: string | null = null
    
    try {
      setRunningTask(taskName)
      const res = await axios.post(`${API_BASE}/tasks/${taskName}/run`, {}, authHeaders)
      
      if (res.data.success) {
        jobId = res.data.job_id
        toast.success(`Task "${taskName}" queued for execution`)
        
        // Poll for status updates
        const maxAttempts = 120  // 120 * 500ms = 60 seconds max (tasks can take longer)
        let attempts = 0
        
        const pollStatus = async () => {
          if (!jobId) return
          
          try {
            const statusRes = await axios.get(`${API_BASE}/tasks/${taskName}/status`, authHeaders)
            const status = statusRes.data
            
            console.log('Task status:', status.status, status)
            
            if (status.status === 'completed') {
              toast.success(`Task "${taskName}" completed successfully`)
            } else if (status.status === 'failed') {
              toast.error(`Task "${taskName}" failed: ${status.error || 'Unknown error'}`)
            }
            // If still queued or running, continue polling
          } catch (err) {
            console.error('Status poll error:', err)
          }
        }
        
        // Start polling
        const pollInterval = setInterval(async () => {
          attempts++
          await pollStatus()
          
          // Get current status to check if we should stop
          try {
            const statusRes = await axios.get(`${API_BASE}/tasks/${taskName}/status`, authHeaders)
            if (statusRes.data.status === 'completed' || statusRes.data.status === 'failed') {
              clearInterval(pollInterval)
              setRunningTask(null)
            } else if (attempts >= maxAttempts) {
              clearInterval(pollInterval)
              setRunningTask(null)
              // Don't show error - just stop polling after timeout
            }
          } catch {
            clearInterval(pollInterval)
            setRunningTask(null)
          }
        }, 500)
        
        // Clean up on unmount
        return () => clearInterval(pollInterval)
      } else {
        toast.error(res.data.message || 'Failed to run task')
      }
    } catch (err: any) {
      console.error('Failed to run task:', err)
      toast.error(err.response?.data?.detail || 'Failed to run task')
    } finally {
      // Only clear if not polling (will be cleared by poll interval)
      if (!jobId) {
        setRunningTask(null)
      }
    }
  }

  const getStatusBadge = (status: string) => {
    const variantMap: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      scheduled: 'secondary',
      pending: 'outline',
      running: 'default',
      disabled: 'secondary',
      error: 'destructive',
      unknown: 'secondary',
    }
    return (
      <Badge variant={variantMap[status] || 'secondary'}>
        {status.toUpperCase()}
      </Badge>
    )
  }

  if (loading) {
    return <div className="p-4">Loading tasks...</div>
  }

  if (error) {
    return (
      <div className="p-4">
        <p className="text-destructive mb-2">{error}</p>
        <Button variant="outline" onClick={fetchTasks}>Retry</Button>
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="flex items-center gap-2 text-xl font-semibold">
          <Clock className="w-6 h-6" />
          Scheduled Tasks
        </h2>
        <div className="flex gap-2">
          {(onCreateNew || onRequestCreate) && (
            <Button onClick={() => { if (onCreateNew) onCreateNew(); if (onRequestCreate) onRequestCreate(); }}>
              <Plus className="w-4 h-4 mr-2" />
              Add New
            </Button>
          )}
          <Button variant="outline" onClick={fetchTasks}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {tasks.length === 0 ? (
        <div className="p-8 text-center text-muted-foreground">
          <p>No tasks found.</p>
          <p>Create a task file in <code>bot/config/tasks/</code></p>
        </div>
      ) : (
        <div className="grid gap-4">
          {tasks.map((task) => (
            <Card key={task.name} className={expandedTask === task.name ? 'border-primary' : ''}>
              {/* Card Header - Click to expand */}
              <CardHeader 
                className="cursor-pointer py-4"
                onClick={() => setExpandedTask(expandedTask === task.name ? null : task.name)}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      {/* Expand icon */}
                      {expandedTask === task.name ? (
                        <ChevronDown className="w-5 h-5 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="w-5 h-5 text-muted-foreground" />
                      )}
                      <CardTitle className="text-lg">{task.name}</CardTitle>
                      {/* Status badge in header */}
                      {getStatusBadge(task.status)}
                    </div>
                    {task.description && (
                      <p className="text-sm text-muted-foreground ml-8 mb-2">
                        {task.description}
                      </p>
                    )}
                    {/* Schedule in collapsed view */}
                    {task.schedule && (
                      <div className="ml-8 flex items-center gap-2 text-sm text-muted-foreground font-mono">
                        <Clock className="w-4 h-4" />
                        {task.schedule}
                      </div>
                    )}
                  </div>
                  {/* Enable toggle in header */}
                  <div onClick={(e) => e.stopPropagation()}>
                    <Button
                      variant={task.enabled ? 'default' : 'secondary'}
                      size="sm"
                      onClick={() => toggleTask(task.name, task.enabled ?? false)}
                      disabled={toggling === task.name}
                    >
                      {(task.enabled ?? false) ? 'ON' : 'OFF'}
                    </Button>
                  </div>
                </div>
              </CardHeader>

              {/* Expanded content - Configuration and action buttons */}
              {expandedTask === task.name && (
                <CardContent className="border-t pt-4 bg-muted/50">
                  <h4 className="text-sm text-muted-foreground mb-2">
                    Configuration
                  </h4>
                  <ScrollArea className="h-[300px] rounded-md bg-background p-4">
                    <pre className="text-sm">
                      {JSON.stringify(tasks.find(t => t.name === task.name), null, 2)}
                    </pre>
                  </ScrollArea>
                  
                  {/* Action buttons at bottom-left of expanded area */}
                  <div className="mt-4 flex gap-2">
                    {/* Run Now button */}
                    <Button 
                      variant="default"
                      onClick={() => runTask(task.name)}
                      disabled={runningTask === task.name}
                    >
                      <Play className="w-4 h-4 mr-2" />
                      {runningTask === task.name ? 'Running...' : 'Run Now'}
                    </Button>
                    
                    {/* Edit button */}
                    {onSelectTask && (
                      <Button 
                        variant="outline"
                        onClick={() => onSelectTask(task.name)}
                      >
                        <Pencil className="w-4 h-4 mr-2" />
                        Edit
                      </Button>
                    )}
                    {onRequestEdit && (
                      <Button 
                        variant="outline"
                        onClick={() => onRequestEdit(task.name)}
                      >
                        <Pencil className="w-4 h-4 mr-2" />
                        Edit
                      </Button>
                    )}
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}