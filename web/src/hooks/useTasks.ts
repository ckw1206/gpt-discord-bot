import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/api'

export interface Task {
  name: string
  config: Record<string, unknown>
}

export interface TaskStatus {
  name: string
  status: 'idle' | 'running' | 'completed' | 'failed'
  last_run?: string
  result?: string
}

export function useTasks(token: string | null) {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = token
    ? { headers: { Authorization: `Bearer ${token}` } }
    : {}

  const fetchTasks = useCallback(async () => {
    if (!token) return

    try {
      setLoading(true)
      const res = await axios.get<Task[]>(`${API_BASE}/tasks`, authHeaders)
      setTasks(res.data || [])
      setError(null)
    } catch (err) {
      setError('Failed to fetch tasks')
      console.error('Failed to fetch tasks:', err)
    } finally {
      setLoading(false)
    }
  }, [token])

  const fetchTask = useCallback(async (name: string) => {
    if (!token) return null

    try {
      const res = await axios.get<Task>(`${API_BASE}/tasks/${name}`, authHeaders)
      return res.data
    } catch (err) {
      console.error('Failed to fetch task:', err)
      return null
    }
  }, [token])

  const createTask = async (name: string, config: Record<string, unknown>) => {
    if (!token) return

    try {
      await axios.post(`${API_BASE}/tasks`, { name, config }, authHeaders)
      await fetchTasks()
    } catch (err) {
      throw err
    }
  }

  const updateTask = async (name: string, config: Record<string, unknown>) => {
    if (!token) return

    try {
      await axios.put(`${API_BASE}/tasks/${name}`, { config }, authHeaders)
      await fetchTasks()
    } catch (err) {
      throw err
    }
  }

  const deleteTask = async (name: string) => {
    if (!token) return

    try {
      await axios.delete(`${API_BASE}/tasks/${name}`, authHeaders)
      await fetchTasks()
    } catch (err) {
      throw err
    }
  }

  const runTask = async (name: string) => {
    if (!token) return

    try {
      await axios.post(`${API_BASE}/tasks/${name}/run`, {}, authHeaders)
    } catch (err) {
      throw err
    }
  }

  const getTaskStatus = useCallback(async (name: string) => {
    if (!token) return null

    try {
      const res = await axios.get<TaskStatus>(`${API_BASE}/tasks/${name}/status`, authHeaders)
      return res.data
    } catch (err) {
      console.error('Failed to get task status:', err)
      return null
    }
  }, [token])

  const reloadTask = async (name: string) => {
    if (!token) return

    try {
      await axios.post(`${API_BASE}/tasks/${name}/reload`, {}, authHeaders)
    } catch (err) {
      throw err
    }
  }

  return {
    tasks,
    loading,
    error,
    fetchTasks,
    fetchTask,
    createTask,
    updateTask,
    deleteTask,
    runTask,
    getTaskStatus,
    reloadTask,
  }
}