import { useState, useEffect } from 'react'
import axios from 'axios'
import { RefreshCw, ChevronDown, ChevronUp, X } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from './ui/table'

const API_BASE = '/api'

interface Server {
  id: string
  name: string
  icon: string | null
  member_count: number
  channel_count: number
  owner_id: string | null
}

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

interface ServerListProps {
  token: string
}

export default function ServerList({ token }: ServerListProps) {
  const [servers, setServers] = useState<Server[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedServerId, setExpandedServerId] = useState<string | null>(null)
  const [serverDetails, setServerDetails] = useState<Record<string, ServerDetail>>({})
  const [detailLoading, setDetailLoading] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'members' | 'channels' | 'permissions'>('members')
  const [permissionMessage, setPermissionMessage] = useState<string | null>(null)

  const authHeaders = {
    headers: { Authorization: `Bearer ${token}` }
  }

  useEffect(() => {
    fetchServers()
  }, [])

  const fetchServers = async () => {
    try {
      const res = await axios.get(`${API_BASE}/servers`, authHeaders)
      setServers(res.data.servers || [])
    } catch (err) {
      console.error('Failed to fetch servers:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchServerDetail = async (serverId: string) => {
    if (serverDetails[serverId]) return // Already loaded
    
    setDetailLoading(serverId)
    try {
      const res = await axios.get(`${API_BASE}/servers/${serverId}`, authHeaders)
      setServerDetails(prev => ({ ...prev, [serverId]: res.data }))
    } catch (err) {
      console.error('Failed to fetch server details:', err)
    } finally {
      setDetailLoading(null)
    }
  }

  const handleServerClick = (serverId: string) => {
    if (expandedServerId === serverId) {
      // Collapse if already expanded
      setExpandedServerId(null)
    } else {
      // Expand and fetch details
      setExpandedServerId(serverId)
      fetchServerDetail(serverId)
      setActiveTab('members')
    }
  }

  const updatePermission = async (serverId: string, permission: string, granted: boolean) => {
    try {
      const res = await axios.put(
        `${API_BASE}/servers/${serverId}/permissions`,
        { [permission]: granted },
        authHeaders
      )
      setServerDetails(prev => ({
        ...prev,
        [serverId]: { ...prev[serverId], permissions: res.data.permissions }
      }))
      setPermissionMessage(`${permission} ${granted ? 'granted' : 'revoked'} successfully`)
      setTimeout(() => setPermissionMessage(null), 3000)
    } catch (err: any) {
      setPermissionMessage(`Error: ${err.response?.data?.detail || 'Failed to update permission'}`)
      setTimeout(() => setPermissionMessage(null), 3000)
    }
  }

  const closeExpanded = () => {
    setExpandedServerId(null)
  }

  if (loading) {
    return <div className="p-4">Loading...</div>
  }

  const expandedServer = expandedServerId ? serverDetails[expandedServerId] : null

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Servers</h2>
        <Button variant="outline" onClick={fetchServers}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {servers.length === 0 ? (
        <p className="text-muted-foreground">No servers found. The bot may not be in any servers yet.</p>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {servers.map((server) => (
            <div key={server.id}>
              <Card 
                className={`cursor-pointer transition-colors hover:bg-muted/50 hover:border-primary ${
                  expandedServerId === server.id ? 'border-primary' : ''
                }`}
                onClick={() => handleServerClick(server.id)}
              >
                <CardContent className="flex flex-row items-center justify-between p-4">
                  <div className="flex items-center gap-4">
                    <div 
                      className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground text-xl font-bold"
                    >
                      {server.icon ? (
                        <img 
                          src={server.icon} 
                          alt={server.name}
                          className="w-12 h-12 rounded-full"
                        />
                      ) : (
                        server.name.charAt(0).toUpperCase()
                      )}
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold">{server.name}</h3>
                      <div className="text-sm text-muted-foreground">
                        <span>Members: {server.member_count}</span>
                        <span className="mx-2">|</span>
                        <span>Channels: {server.channel_count}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {expandedServerId === server.id ? (
                      <ChevronUp className="w-5 h-5 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-muted-foreground" />
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Expanded Content */}
              {expandedServerId === server.id && (
                <div className="mt-2 p-4 bg-muted/30 rounded-md">
                  {detailLoading === server.id ? (
                    <div className="p-4">Loading details...</div>
                  ) : expandedServer ? (
                    <>
                      {/* Tabs */}
                      <div className="flex gap-2 mb-4">
                        {(['members', 'channels', 'permissions'] as const).map((tab) => (
                          <Button
                            key={tab}
                            variant={activeTab === tab ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setActiveTab(tab)}
                            className="capitalize"
                          >
                            {tab}
                          </Button>
                        ))}
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          onClick={closeExpanded}
                          className="ml-auto"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>

                      {permissionMessage && (
                        <div className="mb-2 p-2 text-sm bg-primary/20 rounded">
                          {permissionMessage}
                        </div>
                      )}

                      {/* Members Tab */}
                      {activeTab === 'members' && (
                        <div>
                          {expandedServer.members.length === 0 ? (
                            <p className="text-muted-foreground">No members found</p>
                          ) : (
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>ID</TableHead>
                                  <TableHead>Username</TableHead>
                                  <TableHead>Display Name</TableHead>
                                  <TableHead>Role</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {expandedServer.members.map((member) => (
                                  <TableRow key={member.id}>
                                    <TableCell className="text-muted-foreground">{member.id}</TableCell>
                                    <TableCell>{member.username}</TableCell>
                                    <TableCell>{member.display_name || '-'}</TableCell>
                                    <TableCell>
                                      {member.is_owner && (
                                        <span className="text-xs bg-primary/20 px-2 py-1 rounded">Owner</span>
                                      )}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          )}
                        </div>
                      )}

                      {/* Channels Tab */}
                      {activeTab === 'channels' && (
                        <div>
                          {expandedServer.channels.length === 0 ? (
                            <p className="text-muted-foreground">No channels found</p>
                          ) : (
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Type</TableHead>
                                  <TableHead>ID</TableHead>
                                  <TableHead>Name</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {expandedServer.channels.map((channel) => (
                                  <TableRow key={channel.id}>
                                    <TableCell className="text-xs uppercase">{channel.type}</TableCell>
                                    <TableCell className="text-xs text-muted-foreground">{channel.id}</TableCell>
                                    <TableCell># {channel.name}</TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          )}
                        </div>
                      )}

                      {/* Permissions Tab */}
                      {activeTab === 'permissions' && (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Permission</TableHead>
                              <TableHead>Status</TableHead>
                              <TableHead>Actions</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {Object.entries(expandedServer.permissions).map(([perm, enabled]) => (
                              <TableRow key={perm}>
                                <TableCell className="capitalize">{perm.replace(/_/g, ' ')}</TableCell>
                                <TableCell>
                                  <span className={enabled ? 'text-green-500' : 'text-red-500'}>
                                    {enabled ? '✓ Allowed' : '✗ Denied'}
                                  </span>
                                </TableCell>
                                <TableCell>
                                  <div className="flex gap-2">
                                    <Button
                                      variant={enabled ? 'default' : 'outline'}
                                      size="sm"
                                      onClick={() => updatePermission(server.id, perm, true)}
                                    >
                                      Allow
                                    </Button>
                                    <Button
                                      variant={!enabled ? 'destructive' : 'outline'}
                                      size="sm"
                                      onClick={() => updatePermission(server.id, perm, false)}
                                    >
                                      Deny
                                    </Button>
                                  </div>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      )}
                    </>
                  ) : (
                    <p className="text-muted-foreground">Failed to load details</p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}