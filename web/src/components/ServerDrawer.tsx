import { useState, useEffect } from 'react'
import axios from 'axios'
import { X } from 'lucide-react'
import { Button } from './ui/button'

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

interface ServerDrawerProps {
  serverId: string | null
  token: string
  onClose: () => void
}

export default function ServerDrawer({ serverId, token, onClose }: ServerDrawerProps) {
  const [server, setServer] = useState<ServerDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'members' | 'channels' | 'permissions'>('members')
  const [permissionMessage, setPermissionMessage] = useState<string | null>(null)

  useEffect(() => {
    if (serverId && token) {
      fetchServerDetail(serverId)
    }
  }, [serverId, token])

  const fetchServerDetail = async (guildId: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.get(`${API_BASE}/servers/${guildId}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      setServer(res.data)
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('Server not found (404). The bot may not be in this server.')
      } else {
        setError('Failed to load server details.')
      }
    } finally {
      setLoading(false)
    }
  }

  const updatePermission = async (permission: string, granted: boolean) => {
    if (!serverId || !server) return
    
    try {
      const res = await axios.put(
        `${API_BASE}/servers/${serverId}/permissions`,
        { [permission]: granted },
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setServer({ ...server, permissions: res.data.permissions })
      setPermissionMessage(`${permission} ${granted ? 'granted' : 'revoked'} successfully`)
      setTimeout(() => setPermissionMessage(null), 3000)
    } catch (err: any) {
      setPermissionMessage(`Error: ${err.response?.data?.detail || 'Failed to update permission'}`)
      setTimeout(() => setPermissionMessage(null), 3000)
    }
  }

  if (!serverId) return null

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-black/30 z-[999]"
      />

      {/* Drawer */}
      <div
        className="fixed top-0 right-0 w-[500px] max-w-full h-screen bg-background shadow-[-4px_0_20px_rgba(0,0,0,0.3)] z-[1000] flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div
          className="flex justify-between items-center p-4 border-b"
        >
          <h2 className="m-0 text-xl">
            {server?.name || (loading ? 'Loading...' : 'Server Details')}
          </h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {loading && <div>Loading...</div>}
          
          {error && (
            <div className="text-red-400 p-4">
              {error}
            </div>
          )}

          {server && !loading && !error && (
            <>
              {/* Server Info */}
              <div className="mb-4 flex items-center gap-4">
                {server.icon ? (
                  <img
                    src={server.icon}
                    alt={server.name}
                    className="w-12 h-12 rounded-full"
                  />
                ) : (
                  <div
                    className="w-12 h-12 rounded-full bg-primary flex items-center justify-content text-xl"
                  >
                    {server.name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div>
                  <p className="m-0"><strong>Members:</strong> {server.member_count}</p>
                  <p className="m-0"><strong>Channels:</strong> {server.channel_count}</p>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-2 mb-4">
                {(['members', 'channels', 'permissions'] as const).map((tab) => (
                  <Button
                    key={tab}
                    variant={activeTab === tab ? 'default' : 'outline'}
                    onClick={() => setActiveTab(tab)}
                    className="capitalize"
                  >
                    {tab}
                  </Button>
                ))}
              </div>

              {/* Tab Content */}
              {activeTab === 'members' && (
                <div className="max-h-[300px] overflow-auto">
                  {server.members.length === 0 ? (
                    <p>No members found</p>
                  ) : (
                    <ul className="list-none p-0 m-0">
                      {server.members.slice(0, 50).map((member) => (
                        <li
                          key={member.id}
                          className="p-2 border-b flex justify-between"
                        >
                          <span>
                            {member.display_name || member.username}
                            {member.is_owner && <span className="text-yellow-500"> 👑</span>}
                          </span>
                          <span className="text-muted-foreground text-sm">
                            @{member.username}
                          </span>
                        </li>
                      ))}
                      {server.members.length > 50 && (
                        <li className="p-2 text-muted-foreground">
                          ...and {server.members.length - 50} more
                        </li>
                      )}
                    </ul>
                  )}
                </div>
              )}

              {activeTab === 'channels' && (
                <div className="max-h-[300px] overflow-auto">
                  {server.channels.length === 0 ? (
                    <p>No channels found</p>
                  ) : (
                    <ul className="list-none p-0 m-0">
                      {server.channels.map((channel) => (
                        <li
                          key={channel.id}
                          className="p-2 border-b"
                        >
                          <span className="text-muted-foreground mr-2">
                            #{channel.id}
                          </span>
                          {channel.name}
                          <span className="text-muted-foreground text-sm ml-2">
                            ({channel.type})
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {activeTab === 'permissions' && (
                <div>
                  {permissionMessage && (
                    <div
                      className="p-2 mb-4 rounded"
                      style={{
                        backgroundColor: permissionMessage.startsWith('Error') ? '#ff6b6b22' : '#22c55e22',
                        color: permissionMessage.startsWith('Error') ? '#ff6b6b' : '#22c55e',
                      }}
                    >
                      {permissionMessage}
                    </div>
                  )}
                  <div className="grid grid-cols-[1fr_auto] gap-2">
                    {Object.entries(server.permissions).map(([perm, value]) => (
                      <div
                        key={perm}
                        className="flex justify-between items-center p-2 bg-secondary rounded"
                      >
                        <span className="capitalize text-sm">
                          {perm.replace(/_/g, ' ')}
                        </span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => updatePermission(perm, !value)}
                          className={value ? "text-green-500" : "text-muted-foreground"}
                        >
                          {value ? 'Revoke' : 'Grant'}
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}