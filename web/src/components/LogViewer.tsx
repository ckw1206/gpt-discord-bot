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
  // New fields from structured logging (per logging-guide skill)
  service?: string
  environment?: string
  trace_id?: string
  request_id?: string
  user_id?: string
  // Additional metadata stored in extra_data
  metadata?: Record<string, unknown>
}

interface LogViewerProps {
  token: string
}

// Time range options for filtering logs
type TimeRange = 'all' | '15m' | '1h' | '6h' | '24h' | '7d'

// Helper to load filter from localStorage
const loadFilterFromStorage = (key: string, defaultValue: string): string => {
  if (typeof window === 'undefined') return defaultValue
  const saved = localStorage.getItem(`logViewer_${key}`)
  return saved || defaultValue
}

// Helper to save filter to localStorage
const saveFilterToStorage = (key: string, value: string): void => {
  if (typeof window === 'undefined') return
  localStorage.setItem(`logViewer_${key}`, value)
}

// Filter change handlers that persist to localStorage
const createFilterHandler = (setter: (value: string) => void, storageKey: string) => 
  (value: string) => {
    saveFilterToStorage(storageKey, value)
    setter(value)
  }

export default function LogViewer({ token }: LogViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [levelFilter, setLevelFilter] = useState<string>(() => loadFilterFromStorage('level', 'ALL'))
  const [searchFilter, setSearchFilter] = useState<string>('')
  const [loggerFilter, setLoggerFilter] = useState<string>(() => loadFilterFromStorage('logger', 'ALL'))
  const [serviceFilter, setServiceFilter] = useState<string>(() => loadFilterFromStorage('service', 'ALL'))
  const [environmentFilter, setEnvironmentFilter] = useState<string>(() => loadFilterFromStorage('environment', 'ALL'))
  const [timeRange, setTimeRange] = useState<TimeRange>(() => loadFilterFromStorage('timeRange', 'all') as TimeRange)
  const [availableLoggers, setAvailableLoggers] = useState<string[]>([])
  const [availableServices, setAvailableServices] = useState<string[]>([])
  const [availableLevels, setAvailableLevels] = useState<string[]>([])
  const [expandedLogId, setExpandedLogId] = useState<number | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)
  
  // Refs to track current filter values (avoid stale closures)
  const filtersRef = useRef({ levelFilter, loggerFilter, serviceFilter, environmentFilter, timeRange })
  filtersRef.current = { levelFilter, loggerFilter, serviceFilter, environmentFilter, timeRange }

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  // Fetch available log levels and loggers on mount
  useEffect(() => {
    const fetchFilters = async () => {
      try {
        // Fetch levels from config
        const levelsRes = await axios.get(`${API_BASE}/logs/levels`, authHeaders)
        setAvailableLevels(levelsRes.data || [])
        
        // Fetch loggers from database (all unique loggers, not filtered)
        const loggersRes = await axios.get(`${API_BASE}/logs/types`, authHeaders)
        setAvailableLoggers(['ALL', ...(loggersRes.data || []).sort()])
      } catch (err) {
        console.error('Failed to fetch filter options:', err)
        // Fallback to default levels if API fails
        setAvailableLevels(['INFO', 'WARNING', 'ERROR'])
        setAvailableLoggers(['ALL'])
      }
    }
    fetchFilters()
  }, [token])

  // Calculate "since" parameter based on time range
  const calculateSinceParam = (range: TimeRange): string | undefined => {
    if (range === 'all') return undefined
    
    const now = new Date()
    let since: Date
    
    switch (range) {
      case '15m':
        since = new Date(now.getTime() - 15 * 60 * 1000)
        break
      case '1h':
        since = new Date(now.getTime() - 60 * 60 * 1000)
        break
      case '6h':
        since = new Date(now.getTime() - 6 * 60 * 60 * 1000)
        break
      case '24h':
        since = new Date(now.getTime() - 24 * 60 * 60 * 1000)
        break
      case '7d':
        since = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
        break
      default:
        return undefined
    }
    
    return since.toISOString()
  }

  // Fetch logs when filter changes
  useEffect(() => {
    fetchLogs()
  }, [levelFilter, loggerFilter, serviceFilter, environmentFilter, timeRange])

  // Client-side filtered and sorted logs
  const filteredLogs = (() => {
    try {
      let filtered = logs
      
      // Apply search filter
      if (searchFilter) {
        filtered = filtered.filter(log => 
          (log.message || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
          (log.logger || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
          (log.service || '').toLowerCase().includes(searchFilter.toLowerCase())
        )
      }
      
      // Apply service filter
      if (serviceFilter !== 'ALL') {
        filtered = filtered.filter(log => log.service === serviceFilter)
      }
      
      // Apply environment filter
      if (environmentFilter !== 'ALL') {
        filtered = filtered.filter(log => log.environment === environmentFilter)
      }
      
      // Sort by timestamp ascending (oldest first, newest at bottom)
      return filtered.sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      )
    } catch {
      return logs
    }
  })()

  const fetchLogs = async () => {
    // Use refs to get current filter values (avoid stale closure)
    const { levelFilter, loggerFilter, serviceFilter, environmentFilter, timeRange } = filtersRef.current
    try {
      const params: Record<string, string> = {}
      if (levelFilter !== 'ALL') params.level = levelFilter
      if (loggerFilter !== 'ALL') params.logger = loggerFilter
      if (serviceFilter !== 'ALL') params.service = serviceFilter
      if (environmentFilter !== 'ALL') params.environment = environmentFilter
      
      // Add time range filter (since parameter)
      const sinceParam = calculateSinceParam(timeRange)
      if (sinceParam) {
        params.since = sinceParam
      }
      
      const res = await axios.get(`${API_BASE}/logs`, { ...authHeaders, params })
      setLogs(res.data.logs || [])
      
      // Extract unique services for the filter dropdown
      const services = new Set<string>(
        (res.data.logs?.map((log: LogEntry) => log.service) || []).filter(Boolean)
      )
      setAvailableServices(['ALL', ...Array.from(services).sort()])
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
        // Append new logs at the end (newest at bottom), keep last 500
        setLogs(prev => [...prev, log].slice(-500))
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
        
        {/* Level filter - only show levels that are enabled in config */}
        <Select value={levelFilter} onValueChange={createFilterHandler(setLevelFilter, 'level')}>
          <SelectTrigger className="w-[120px] h-8 border border-input rounded-md px-3 text-sm">
            <SelectValue placeholder="Level" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ALL">All Levels</SelectItem>
            {availableLevels.map(level => (
              <SelectItem key={level} value={level}>
                {level.charAt(0) + level.slice(1).toLowerCase()}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        {/* Logger filter */}
        <Select value={loggerFilter} onValueChange={createFilterHandler(setLoggerFilter, 'logger')}>
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
        
        {/* Service filter (new field from structured logging) */}
        <Select value={serviceFilter} onValueChange={createFilterHandler(setServiceFilter, 'service')}>
          <SelectTrigger className="w-[130px] h-8 border border-input rounded-md px-3 text-sm">
            <SelectValue placeholder="Service" />
          </SelectTrigger>
          <SelectContent>
            {availableServices.map(service => (
              <SelectItem key={service} value={service}>
                {service === 'ALL' ? 'All Services' : service}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        
        {/* Environment filter (new field from structured logging) */}
        <Select value={environmentFilter} onValueChange={createFilterHandler(setEnvironmentFilter, 'environment')}>
          <SelectTrigger className="w-[130px] h-8 border border-input rounded-md px-3 text-sm">
            <SelectValue placeholder="Environment" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ALL">All Envs</SelectItem>
            <SelectItem value="development">Development</SelectItem>
            <SelectItem value="production">Production</SelectItem>
          </SelectContent>
        </Select>
        
        {/* Time range filter */}
        <Select value={timeRange} onValueChange={createFilterHandler(setTimeRange as (value: string) => void, 'timeRange') as (value: string) => void}>
          <SelectTrigger className="w-[100px] h-8 border border-input rounded-md px-3 text-sm">
            <SelectValue placeholder="Time" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Time</SelectItem>
            <SelectItem value="15m">Last 15 min</SelectItem>
            <SelectItem value="1h">Last 1 hour</SelectItem>
            <SelectItem value="6h">Last 6 hours</SelectItem>
            <SelectItem value="24h">Last 24 hours</SelectItem>
            <SelectItem value="7d">Last 7 days</SelectItem>
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
            {/* Main log line */}
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-muted-foreground font-mono">
                {/* Timestamp includes local timezone offset (e.g., +08:00), parse and display as-is */}
                {new Date(log.timestamp).toISOString().replace('Z', '')}
              </span>
              <span className={`font-bold ${getLevelColor(log.level)}`}>
                [{log.level}]
              </span>
              {log.service && (
                <span className="text-green-500">[{log.service}]</span>
              )}
              {log.environment && (
                <span className={`text-xs ${log.environment === 'production' ? 'text-red-400' : 'text-blue-400'}`}>
                  [{log.environment}]
                </span>
              )}
              <span className="text-muted-foreground">[{log.logger}]</span>
              <span className="text-foreground/80">{log.message}</span>
              
              {/* Expand button for additional fields */}
              {(log.trace_id || log.request_id || log.user_id || log.metadata) && (
                <button
                  onClick={() => setExpandedLogId(expandedLogId === log.id ? null : log.id)}
                  className="text-xs text-muted-foreground hover:text-foreground ml-2"
                >
                  {expandedLogId === log.id ? '▼' : '▶'}
                </button>
              )}
            </div>
            
            {/* Expanded details */}
            {expandedLogId === log.id && (
              <div className="ml-4 mt-1 p-2 bg-muted/50 rounded text-xs text-muted-foreground">
                {log.trace_id && <div><span className="text-cyan-400">trace_id:</span> {log.trace_id}</div>}
                {log.request_id && <div><span className="text-cyan-400">request_id:</span> {log.request_id}</div>}
                {log.user_id && <div><span className="text-cyan-400">user_id:</span> {log.user_id}</div>}
                {log.metadata && (
                  <div>
                    <span className="text-cyan-400">metadata:</span>
                    <pre className="mt-1 text-xs">{JSON.stringify(log.metadata, null, 2)}</pre>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={logsEndRef} />
        {filteredLogs.length === 0 && <p className="text-muted-foreground">No logs available</p>}
      </ScrollArea>
    </div>
  )
}