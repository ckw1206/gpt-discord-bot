import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export interface Model {
  name: string
  persona: string
  supports_tools: boolean
  tools: string[]
  think: boolean
}

export function useModels(token: string | null) {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = token
    ? { headers: { Authorization: `Bearer ${token}` } }
    : {}

  const fetchModels = useCallback(async () => {
    if (!token) return

    try {
      setLoading(true)
      const res = await axios.get<{ models: Model[] }>(`${API_BASE}/config`, authHeaders)
      setModels(res.data.models || [])
      setError(null)
    } catch (err) {
      setError('Failed to fetch models')
      console.error('Failed to fetch models:', err)
    } finally {
      setLoading(false)
    }
  }, [token])

  const addModel = async (model: Model) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config/models`, {
        action: 'add',
        model_name: model.name,
        data: {
          persona: model.persona,
          supports_tools: model.supports_tools,
          tools: model.tools,
          think: model.think
        }
      }, authHeaders)
      await fetchModels()
    } catch (err) {
      throw err
    }
  }

  const updateModel = async (name: string, data: Partial<Model>) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config/models`, {
        action: 'update',
        model_name: name,
        data: {
          persona: data.persona,
          supports_tools: data.supports_tools,
          tools: data.tools,
          think: data.think
        }
      }, authHeaders)
      await fetchModels()
    } catch (err) {
      throw err
    }
  }

  const deleteModel = async (name: string) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/config/models`, {
        action: 'delete',
        model_name: name
      }, authHeaders)
      await fetchModels()
    } catch (err) {
      throw err
    }
  }

  return {
    models,
    loading,
    error,
    fetchModels,
    addModel,
    updateModel,
    deleteModel,
  }
}