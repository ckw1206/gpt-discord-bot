import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { ArrowLeft } from 'lucide-react'
import { Button } from './ui/button'
import { Badge } from './ui/badge'

const API_BASE = '/api'

interface GuildMember {
  id: number
  username: string
  display_name: string | null
  is_owner: boolean
}

interface GuildChannel {
  id: number
  name: string
  type: string
}

interface GuildPermissions {
  can_send_messages: boolean
  can_embed_links: boolean
  can_attach_files: boolean
  can_use_external_emojis: boolean
  can_manage_messages: boolean
  can_manage_channels: boolean
  can_kick_members: boolean
  can_ban_members: boolean
  can_manage_guild: boolean
}

interface ServerDetail {
  id: number
  name: string
  icon: string | null
  owner_id: number | null
  member_count: number
  channel_count: number
  members: GuildMember[]
  channels: GuildChannel[]
  permissions: GuildPermissions
}

interface ServerDetailProps {
  token: string
}

export default function ServerDetail({ token }: ServerDetailProps) {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [server, setServer] = useState<ServerDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'members' | 'channels' | 'permissions'>('members')
  const [permissionMessage, setPermissionMessage] = useState<string | null>(null)

  console.log('ServerDetail: token=', token ? 'exists' : 'NULL', 'id=', id)

  useEffect(() => {
    console.log('useEffect triggered: id=', id, 'token=', token ? 'exists' : 'NULL')
    if (id && token) {
      // Keep as string to avoid JavaScript number precision loss with large guild IDs
      fetchServerDetail(id)
    } else if (!token) {
      console.error('No token in useEffect!')
      setError('Not authenticated. Please login again.')
      setLoading(false)
    }
  }, [id, token])

  const fetchServerDetail = async (guildId: string) => {
    if (!token) {
      console.error('No token available!')
      setError('Not authenticated. Please login again.')
      setLoading(false)
      return
    }
    
    try {
      console.log('Fetching server detail for guildId:', guildId, '(type:', typeof guildId + ')')
      console.log('Using token:', token.substring(0, 20) + '...')
      
      const res = await axios.get(`${API_BASE}/servers/${guildId}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      console.log('Server detail response:', res.data)
      setServer(res.data)
    } catch (err: any) {
      console.error('Failed to fetch server detail:', err)
      console.error('Error status:', err.response?.status)
      console.error('Error response:', err.response?.data)
      console.error('Error message:', err.message)
      
      if (err.response?.status === 404) {
        setError('Server not found (404). The bot may not be in this server.')
      } else if (err.response?.status === 401) {
        setError('Unauthorized. Please login again.')
      } else if (err.response?.status === 403) {
        setError('Forbidden. You do not have access to this server.')
      } else {
        setError(err.response?.data?.detail || err.message || 'Failed to load server details')
      }
    } finally {
      setLoading(false)
    }
  }

  const updatePermission = async (permission: string, action: 'grant' | 'revoke') => {
    if (!id) return
    
    setPermissionMessage(null)
    try {
      const res = await axios.put(
        `${API_BASE}/servers/${id}/permissions`,
        { action, permission },
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setPermissionMessage(res.data.message)
      // Refresh permissions - keep as string to avoid precision loss
      if (id) {
        fetchServerDetail(id)
      }
    } catch (err: any) {
      setPermissionMessage(err.response?.data?.message || 'Failed to update permission')
    }
  }

  if (loading) {
    return <div className="p-8">Loading...</div>
  }

  if (error) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={() => navigate('/')}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
        <p className="text-red-500 mt-4">{error}</p>
      </div>
    )
  }

  if (!server) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={() => navigate('/')}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
        <p>Server not found</p>
      </div>
    )
  }

  const permissionList = [
    { key: 'can_send_messages', label: 'Send Messages', discord: 'send_messages' },
    { key: 'can_embed_links', label: 'Embed Links', discord: 'embed_links' },
    { key: 'can_attach_files', label: 'Attach Files', discord: 'attach_files' },
    { key: 'can_use_external_emojis', label: 'External Emoji', discord: 'external_emojis' },
    { key: 'can_manage_messages', label: 'Manage Messages', discord: 'manage_messages' },
    { key: 'can_manage_channels', label: 'Manage Channels', discord: 'manage_channels' },
    { key: 'can_kick_members', label: 'Kick Members', discord: 'kick_members' },
    { key: 'can_ban_members', label: 'Ban Members', discord: 'ban_members' },
    { key: 'can_manage_guild', label: 'Manage Server', discord: 'manage_guild' },
  ]

  return (
    <div className="p-4">
      <Button variant="ghost" onClick={() => navigate('/')} className="mb-4">
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to Dashboard
      </Button>
      
      {/* Server Header */}
      <div className="flex items-center gap-4 mb-6">
        <div className="relative">
          {server.icon ? (
            <img 
              src={server.icon} 
              alt={server.name}
              className="w-20 h-20 rounded-full"
            />
          ) : (
            <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center text-muted-foreground text-2xl font-bold">
              {server.name.charAt(0).toUpperCase()}
            </div>
          )}
        </div>
        <div>
          <h2 className="text-2xl font-semibold m-0">{server.name}</h2>
          <p className="text-muted-foreground mt-1">
            {server.member_count} members • {server.channel_count} channels
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4">
        <Button 
          variant={activeTab === 'members' ? 'default' : 'outline'}
          onClick={() => setActiveTab('members')}
        >
          Members ({server.members.length})
        </Button>
        <Button 
          variant={activeTab === 'channels' ? 'default' : 'outline'}
          onClick={() => setActiveTab('channels')}
        >
          Channels ({server.channels.length})
        </Button>
        <Button 
          variant={activeTab === 'permissions' ? 'default' : 'outline'}
          onClick={() => setActiveTab('permissions')}
        >
          Permissions
        </Button>
      </div>

      {/* Members Tab */}
      {activeTab === 'members' && (
        <div className="max-h-[400px] overflow-auto bg-secondary rounded-lg p-4">
          {server.members.length === 0 ? (
            <p>No members found</p>
          ) : (
            <div className="flex flex-col gap-2">
              {server.members.slice(0, 100).map((member) => (
                <div 
                  key={member.id}
                  className="flex justify-between items-center p-2 bg-muted rounded"
                >
                  <div>
                    <span className="font-bold">{member.username}</span>
                    {member.display_name && (
                      <span className="text-muted-foreground ml-2">
                        (aka {member.display_name})
                      </span>
                    )}
                    {member.is_owner && (
                      <Badge variant="secondary" className="ml-2">
                        Owner
                      </Badge>
                    )}
                  </div>
                  <span className="text-muted-foreground text-sm">ID: {member.id}</span>
                </div>
              ))}
              {server.members.length > 100 && (
                <p className="text-muted-foreground text-center">
                  ... and {server.members.length - 100} more members
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Channels Tab */}
      {activeTab === 'channels' && (
        <div className="max-h-[400px] overflow-auto bg-secondary rounded-lg p-4">
          {server.channels.length === 0 ? (
            <p>No channels found</p>
          ) : (
            <div className="flex flex-col gap-2">
              {server.channels.map((channel) => (
                <div 
                  key={channel.id}
                  className="flex justify-between items-center p-2 bg-muted rounded"
                >
                  <div>
                    <span className="font-bold"># {channel.name}</span>
                    <Badge variant="outline" className="ml-2">
                      {channel.type}
                    </Badge>
                  </div>
                  <span className="text-muted-foreground text-sm">ID: {channel.id}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Permissions Tab */}
      {activeTab === 'permissions' && (
        <div className="bg-secondary rounded-lg p-4">
          <h3 className="mt-0">Bot Permissions</h3>
          
          {permissionMessage && (
            <div className="p-3 mb-4 rounded"
              style={{
                backgroundColor: permissionMessage.includes('Error') || permissionMessage.includes('Invalid') || permissionMessage.includes('Failed') ? '#7f1d1d' : '#14532d',
              }}
            >
              {permissionMessage}
            </div>
          )}
          
          <div className="flex flex-col gap-3">
            {permissionList.map((perm) => {
              const hasPermission = server.permissions[perm.key as keyof GuildPermissions]
              return (
                <div 
                  key={perm.key}
                  className="flex justify-between items-center p-3 bg-muted rounded"
                >
                  <div className="flex items-center gap-2">
                    <span 
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: hasPermission ? '#22c55e' : '#6b7280'
                      }}
                    />
                    <span>{perm.label}</span>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-green-500 border-green-500 hover:bg-green-500 hover:text-white"
                      onClick={() => updatePermission(perm.discord, 'grant')}
                    >
                      Grant
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-red-500 border-red-500 hover:bg-red-500 hover:text-white"
                      onClick={() => updatePermission(perm.discord, 'revoke')}
                    >
                      Revoke
                    </Button>
                  </div>
                </div>
              )
            })}
          </div>
          
          <p className="text-muted-foreground text-sm mt-4">
            Note: Permission changes are simulated. Actual Discord permission management requires administrator access.
          </p>
        </div>
      )}
    </div>
  )
}