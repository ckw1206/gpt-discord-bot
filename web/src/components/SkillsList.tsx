import { useState, useEffect } from 'react'
import axios from 'axios'
import { RefreshCw, Code, Info } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'

const API_BASE = '/api'

interface SkillParameter {
  name: string
  type: string
  required: boolean
  description: string
}

interface SkillInfo {
  name: string
  filename: string
  description: string | null
  parameters: SkillParameter[]
}

interface SkillsListProps {
  token: string
  onSelectSkill?: (name: string) => void
}

export default function SkillsList({ token, onSelectSkill }: SkillsListProps) {
  const [skills, setSkills] = useState<SkillInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchSkills()
  }, [])

  const fetchSkills = async () => {
    try {
      setLoading(true)
      const res = await axios.get(`${API_BASE}/skills`, authHeaders)
      setSkills(res.data || [])
      setError(null)
    } catch (err: unknown) {
      console.error('Failed to fetch skills:', err)
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to load skills')
    } finally {
      setLoading(false)
    }
  }

  const handleSkillClick = (name: string) => {
    const newSelection = selectedSkill === name ? null : name
    setSelectedSkill(newSelection)
    if (onSelectSkill) {
      onSelectSkill(newSelection || '')
    }
  }

  if (loading) {
    return <div className="p-4">Loading skills...</div>
  }

  if (error) {
    return (
      <div className="p-4">
        <p className="text-destructive mb-2">{error}</p>
        <Button variant="outline" onClick={fetchSkills}>Retry</Button>
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="flex items-center gap-2 text-xl font-semibold">
          <Code className="w-6 h-6" />
          Skills (Tools)
        </h2>
        <Button variant="outline" onClick={fetchSkills}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      <p className="text-sm text-muted-foreground mb-4">
        Read-only view of available skills. Modification is not in scope (future consideration).
      </p>

      {skills.length === 0 ? (
        <div className="p-8 text-center text-muted-foreground">
          <p>No skills found.</p>
          <p>Skill files should be in <code>bot/llm/tools/skills/</code></p>
        </div>
      ) : (
        <div className="grid gap-4">
          {skills.map((skill) => (
            <Card 
              key={skill.name}
              className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                selectedSkill === skill.name ? 'border-primary' : ''
              }`}
              onClick={() => handleSkillClick(skill.name)}
            >
              <CardHeader className="py-3">
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-lg">{skill.name}</CardTitle>
                    {skill.description && (
                      <p className="text-sm text-muted-foreground mt-1">
                        {skill.description}
                      </p>
                    )}
                  </div>
                  <Info className="w-5 h-5 text-muted-foreground" />
                </div>
              </CardHeader>

              {selectedSkill === skill.name && skill.parameters.length > 0 && (
                <CardContent className="pt-0 border-t">
                  <h4 className="text-sm text-muted-foreground mb-3 mt-3">
                    Parameters
                  </h4>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Required</TableHead>
                        <TableHead>Description</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {skill.parameters.map((param, idx) => (
                        <TableRow key={idx}>
                          <TableCell>
                            <code className="bg-muted px-2 py-1 rounded text-sm">
                              {param.name}
                            </code>
                          </TableCell>
                          <TableCell className="text-blue-400">
                            {param.type}
                          </TableCell>
                          <TableCell>
                            {param.required ? (
                              <span className="text-destructive font-medium">Yes</span>
                            ) : (
                              <span className="text-muted-foreground">No</span>
                            )}
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {param.description}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}