import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export interface Provider {
  name: string
  base_url: string
  api_key: string
}

export function useProviders(token: string | null) {
  const [providers, setProviders] = useState<Provider[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = token
    ? { headers: { Authorization: `Bearer ${token}` } }
    : {}

  const fetchProviders = useCallback(async () => {
    if (!token) return

    try {
      setLoading(true)
      const res = await axios.get<{ providers: Provider[] }>(`${API_BASE}/config`, authHeaders)
      setProviders(res.data.providers || [])
      setError(null)
    } catch (err) {
      setError('Failed to fetch providers')
      console.error('Failed to fetch providers:', err)
    } finally {
      setLoading(false)
    }
  }, [token])

  const addProvider = async (provider: Omit<Provider, 'api_key'> & { api_key?: string }) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config/providers`, {
        action: 'add',
        name: provider.name,
        data: {
          base_url: provider.base_url,
          api_key: provider.api_key || ''
        }
      }, authHeaders)
      await fetchProviders()
    } catch (err) {
      throw err
    }
  }

  const updateProvider = async (name: string, data: Partial<Provider>) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config/providers`, {
        action: 'update',
        name,
        data: {
          base_url: data.base_url,
          api_key: data.api_key
        }
      }, authHeaders)
      await fetchProviders()
    } catch (err) {
      throw err
    }
  }

  const deleteProvider = async (name: string) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config/providers`, {
        action: 'delete',
        name
      }, authHeaders)
      await fetchProviders()
    } catch (err) {
      throw err
    }
  }

  return {
    providers,
    loading,
    error,
    fetchProviders,
    addProvider,
    updateProvider,
    deleteProvider,
  }
}