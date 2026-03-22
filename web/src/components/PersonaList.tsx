import { useState, useEffect } from 'react'
import axios from 'axios'
import { RefreshCw, Plus } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

const API_BASE = '/api'

interface PersonaInfo {
  name: string
  filename: string
  description: string | null
}

interface PersonaListProps {
  token: string
  onSelectPersona?: (name: string) => void
  onCreateNew?: () => void
}

export default function PersonaList({ token, onSelectPersona, onCreateNew }: PersonaListProps) {
  const [personas, setPersonas] = useState<PersonaInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchPersonas()
  }, [])

  const fetchPersonas = async () => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/personas`, authHeaders)
      setPersonas(res.data || [])
      setError(null)
    } catch (err: unknown) {
      console.error('Failed to fetch personas:', err)
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to load personas')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="p-4">Loading personas...</div>
  }

  if (error) {
    return (
      <div className="p-4">
        <p className="text-destructive mb-2">{error}</p>
        <Button variant="outline" onClick={fetchPersonas}>Retry</Button>
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Personas</h2>
        <div className="flex gap-2">
          {onCreateNew && (
            <Button onClick={onCreateNew}>
              <Plus className="w-4 h-4 mr-2" />
              Add New
            </Button>
          )}
          <Button variant="outline" onClick={fetchPersonas}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {personas.length === 0 ? (
        <div className="p-8 text-center text-muted-foreground">
          <p>No personas found.</p>
          <p>Create a persona file in <code>bot/config/personas/</code></p>
        </div>
      ) : (
        <div className="grid gap-4">
          {personas.map((persona) => (
            <Card 
              key={persona.name}
              className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                onSelectPersona ? 'hover:border-primary' : ''
              }`}
              onClick={() => onSelectPersona?.(persona.name)}
            >
              <CardHeader className="py-3">
                <CardTitle className="text-lg">{persona.name}</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                {persona.description && (
                  <p className="text-sm text-muted-foreground">
                    {persona.description}
                  </p>
                )}
                <p className="text-xs text-muted-foreground mt-2">
                  {persona.filename}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Card className="mt-6 bg-muted/50">
        <CardHeader className="py-3">
          <CardTitle className="text-base">Add New Persona</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <p className="text-sm text-muted-foreground">
            Create a new persona file in <code>bot/config/personas/</code> with a <code>.md</code> extension.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}