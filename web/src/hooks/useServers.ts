import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export interface Server {
  guild_id: string
  name: string
  icon_url?: string
}

export function useServers(token: string | null) {
  const [servers, setServers] = useState<Server[]>([])
  const [currentServer, setCurrentServer] = useState<Server | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = token
    ? { headers: { Authorization: `Bearer ${token}` } }
    : {}

  const fetchServers = useCallback(async () => {
    if (!token) return

    try {
      setLoading(true)
      const res = await axios.get<Server[]>(`${API_BASE}/servers`, authHeaders)
      setServers(res.data || [])
      setError(null)
    } catch (err) {
      setError('Failed to fetch servers')
      console.error('Failed to fetch servers:', err)
    } finally {
      setLoading(false)
    }
  }, [token])

  const fetchServer = useCallback(async (guildId: string) => {
    if (!token) return null

    try {
      const res = await axios.get<Server>(`${API_BASE}/servers/${guildId}`, authHeaders)
      setCurrentServer(res.data)
      return res.data
    } catch (err) {
      console.error('Failed to fetch server:', err)
      return null
    }
  }, [token])

  const setSelectedServer = (guildId: string) => {
    const server = servers.find(s => s.guild_id === guildId)
    if (server) {
      setCurrentServer(server)
    }
  }

  return {
    servers,
    currentServer,
    loading,
    error,
    fetchServers,
    fetchServer,
    setSelectedServer,
  }
}