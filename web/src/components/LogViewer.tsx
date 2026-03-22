import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { ScrollArea } from './ui/scroll-area'
import { Badge } from './ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Input } from './ui/input'

const API_BASE = '/api'

interface LogEntry {
  id: number
  timestamp: string
  level: string
  message: string
  logger: string
}

interface LogViewerProps {
  token: string
}

export default function LogViewer({ token }: LogViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [levelFilter, setLevelFilter] = useState<string>('ALL')
  const [searchFilter, setSearchFilter] = useState<string>('')
  const [loggerFilter, setLoggerFilter] = useState<string>('ALL')
  const [availableLoggers, setAvailableLoggers] = useState<string[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Fetch logs when level or logger filter changes
  useEffect(() => {
    fetchLogs()
  }, [levelFilter, loggerFilter])

  // Client-side filtered logs based on search
  const filteredLogs = (() => {
    try {
      if (!searchFilter || !logs || logs.length === 0) return logs
      return logs.filter(log => 
        (log.message || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
        (log.logger || '').toLowerCase().includes(searchFilter.toLowerCase())
      )
    } catch {
      return logs
    }
  })()

  const fetchLogs = async () => {
    try {
      const params: Record<string, string> = {}
      if (levelFilter !== 'ALL') params.level = levelFilter
      if (loggerFilter !== 'ALL') params.logger = loggerFilter
      const res = await axios.get(`${API_BASE}/logs`, { ...authHeaders, params })
      setLogs(res.data.logs || [])
      
      // Extract unique loggers for the filter dropdown (filter out empty strings)
      const loggers = new Set<string>(
        (res.data.logs?.map((log: LogEntry) => log.logger) || []).filter(Boolean)
      )
      setAvailableLoggers(['ALL', ...Array.from(loggers).sort()])
    } catch (err) {
      console.error('Failed to fetch logs:', err)
    }
  }

  // Connect to WebSocket for real-time logs
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/logs`
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
      setWsConnected(true)
    }

    ws.onmessage = (event) => {
      if (event.data === 'pong') return // Ignore ping responses
      
      try {
        const log = JSON.parse(event.data)
        setLogs(prev => [log, ...prev].slice(0, 500)) // Keep last 500 logs
      } catch (err) {
        console.error('Failed to parse log:', err)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setWsConnected(false)
    }

    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
    }

    return () => {
      ws.close()
    }
  }, [])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleLevelChange = (level: string) => {
    setLevelFilter(level)
    fetchLogs()
  }

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'text-destructive'
      case 'WARNING': return 'text-yellow-500'
      case 'INFO': return 'text-blue-400'
      case 'DEBUG': return 'text-purple-400'
      default: return 'text-foreground'
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Header row with title, status badge, and filters */}
      <div className="flex gap-3 items-center mb-2 flex-shrink-0 flex-wrap">
        <span className="font-bold text-lg">Logs</span>
        <Badge variant={wsConnected ? 'default' : 'destructive'}>
          {wsConnected ? 'Live' : 'Disconnected'}
        </Badge>
        
        {/* Search input */}
        <Input
          placeholder="Search logs..."
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          className="w-[200px] h-8"
        />
        
        {/* Level filter */}
        <Select value={levelFilter} onValueChange={handleLevelChange}>
          <SelectTrigger className="w-[120px] h-8 border border-input rounded-md px-3 text-sm">
            <SelectValue placeholder="Level" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ALL">All Levels</SelectItem>
            <SelectItem value="DEBUG">Debug</SelectItem>
            <SelectItem value="INFO">Info</SelectItem>
            <SelectItem value="WARNING">Warning</SelectItem>
            <SelectItem value="ERROR">Error</SelectItem>
          </SelectContent>
        </Select>
        
        {/* Logger filter */}
        <Select value={loggerFilter} onValueChange={setLoggerFilter}>
          <SelectTrigger className="w-[150px] h-8 border border-input rounded-md px-3 text-sm">
            <SelectValue placeholder="Logger" />
          </SelectTrigger>
          <SelectContent>
            {availableLoggers.map(logger => (
              <SelectItem key={logger} value={logger}>
                {logger === 'ALL' ? 'All Loggers' : logger}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        <span className="ml-auto text-sm text-muted-foreground">
          {filteredLogs.length} / {logs.length} entries
        </span>
      </div>

      {/* Log content - fills remaining space */}
      <ScrollArea className="flex-1 rounded-lg bg-background p-2 font-mono text-sm min-h-0">
        {filteredLogs.map((log) => (
          <div key={log.id} className="mb-1">
            <span className="text-muted-foreground mr-2">
              {new Date(log.timestamp).toLocaleTimeString()}
            </span>
            <span className={`font-bold mr-2 min-w-[60px] inline-block ${getLevelColor(log.level)}`}>
              [{log.level}]
            </span>
            <span className="text-muted-foreground mr-2">[{log.logger}]</span>
            <span className="text-foreground/80">{log.message}</span>
          </div>
        ))}
        <div ref={logsEndRef} />
        {filteredLogs.length === 0 && <p className="text-muted-foreground">No logs available</p>}
      </ScrollArea>
    </div>
  )
}