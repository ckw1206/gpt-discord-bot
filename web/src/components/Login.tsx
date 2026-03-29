import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth, toast } from '../App'
import axios from 'axios'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'

const API_BASE = '/api'

export default function Login() {
  const { setToken } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [isSetup, setIsSetup] = useState(false)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  // Check if setup is needed on mount
  useEffect(() => {
    axios.get(`${API_BASE}/auth/has-users`)
      .then(res => {
        setIsSetup(!res.data.has_users)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    
    try {
      const endpoint = isSetup ? '/auth/setup' : '/auth/login'
      const res = await axios.post(`${API_BASE}${endpoint}`, {
        username,
        password,
      })
      
      setToken(res.data.access_token)
      toast.success(isSetup ? 'Admin account created!' : 'Login successful!')
      navigate('/')
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      toast.error(error.response?.data?.detail || 'Authentication failed')
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p>Loading...</p>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <Card className="w-[350px]">
        <CardHeader>
          <CardTitle>{isSetup ? 'Setup Admin Account' : 'Login'}</CardTitle>
          <CardDescription>
            {isSetup 
              ? 'Create your admin account to get started' 
              : 'Enter your credentials to access the dashboard'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <Button 
              type="submit" 
              disabled={submitting}
              className="w-full"
            >
              {submitting ? 'Please wait...' : (isSetup ? 'Create Admin' : 'Login')}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}