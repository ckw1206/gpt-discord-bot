import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export interface Persona {
  name: string
  content: string
}

export interface PersonaDetail extends Persona {
  // Add additional fields based on API response
}

export interface PersonaUsage {
  task_name: string
  run_count: number
}

export function usePersonas(token: string | null) {
  const [personas, setPersonas] = useState<Persona[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = token
    ? { headers: { Authorization: `Bearer ${token}` } }
    : {}

  const fetchPersonas = useCallback(async () => {
    if (!token) return

    try {
      setLoading(true)
      const res = await axios.get<Persona[]>(`${API_BASE}/personas`, authHeaders)
      setPersonas(res.data || [])
      setError(null)
    } catch (err) {
      setError('Failed to fetch personas')
      console.error('Failed to fetch personas:', err)
    } finally {
      setLoading(false)
    }
  }, [token])

  const fetchPersona = useCallback(async (name: string) => {
    if (!token) return null

    try {
      const res = await axios.get<PersonaDetail>(`${API_BASE}/personas/${name}`, authHeaders)
      return res.data
    } catch (err) {
      console.error('Failed to fetch persona:', err)
      return null
    }
  }, [token])

  const createPersona = async (name: string, content: string) => {
    if (!token) return

    try {
      await axios.post(`${API_BASE}/personas`, { name, content }, authHeaders)
      await fetchPersonas()
    } catch (err) {
      throw err
    }
  }

  const updatePersona = async (name: string, content: string) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/personas/${name}`, { content }, authHeaders)
      await fetchPersonas()
    } catch (err) {
      throw err
    }
  }

  const deletePersona = async (name: string) => {
    if (!token) return

    try {
      await axios.delete(`${API_BASE}/personas/${name}`, authHeaders)
      await fetchPersonas()
    } catch (err) {
      throw err
    }
  }

  return {
    personas,
    loading,
    error,
    fetchPersonas,
    fetchPersona,
    createPersona,
    updatePersona,
    deletePersona,
  }
}