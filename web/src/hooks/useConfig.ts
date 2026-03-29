import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export interface Config {
  // Add config fields based on actual API response
  [key: string]: unknown
}

export interface Provider {
  name: string
  base_url: string
  api_key: string
}

export interface Model {
  name: string
  persona: string
  supports_tools: boolean
  tools: string[]
  think: boolean
}

export function useConfig(token: string | null) {
  const [config, setConfig] = useState<Config | null>(null)
  const [providers, setProviders] = useState<Provider[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = token 
    ? { headers: { Authorization: `Bearer ${token}` } }
    : {}

  const fetchConfig = useCallback(async () => {
    if (!token) {
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      const res = await axios.get<Config>(`${API_BASE}/config`, authHeaders)
      setConfig(res.data)
      setProviders(res.data.providers as Provider[] || [])
      setModels(res.data.models as Model[] || [])
      setError(null)
    } catch (err) {
      setError('Failed to fetch config')
      console.error('Failed to fetch config:', err)
    } finally {
      setLoading(false)
    }
  }, [token])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  const updateConfig = async (updates: Record<string, unknown>) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config`, { fields: updates }, authHeaders)
      await fetchConfig()
    } catch (err) {
      throw err
    }
  }

  return {
    config,
    providers,
    models,
    loading,
    error,
    refetch: fetchConfig,
    updateConfig,
  }
}