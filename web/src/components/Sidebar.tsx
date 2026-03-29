import { useState, useEffect } from 'react'
import { 
  Home as HomeIcon, 
  Cog as Cog6ToothIcon, 
  Server as ServerStackIcon, 
  User as UserIcon,
  ListTodo as ListBulletIcon,
  Code as CodeBracketIcon,
  LogOut
} from 'lucide-react'
type TabType = 'dashboard' | 'config' | 'servers' | 'personas' | 'tasks' | 'skills'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
  onLogout: () => void
}

interface NavItem {
  id: TabType
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: HomeIcon },
  { id: 'config', label: 'Config', icon: Cog6ToothIcon },
  { id: 'servers', label: 'Servers', icon: ServerStackIcon },
  { id: 'personas', label: 'Personas', icon: UserIcon },
  { id: 'tasks', label: 'Tasks', icon: ListBulletIcon },
  { id: 'skills', label: 'Skills', icon: CodeBracketIcon },
]

export default function Sidebar({ activeTab, onTabChange, onLogout }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Responsive: collapse to icons on narrow screens
  useEffect(() => {
    const handleResize = () => {
      setIsCollapsed(window.innerWidth <= 768)
    }

    handleResize() // Check on mount
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <aside
      className={`flex flex-col h-screen fixed left-0 top-0 border-r bg-card text-card-foreground transition-all duration-200 ${
        isCollapsed ? 'w-[60px] min-w-[60px]' : 'w-[200px] min-w-[200px]'
      }`}
    >
      {/* Logo/Title */}
      <div className="p-2 mb-4 border-b">
        {!isCollapsed && (
          <h2 className="text-lg font-semibold">
            GPT Discord Bot
          </h2>
        )}
        {isCollapsed && (
          <div className="flex justify-center">
            <Cog6ToothIcon className="w-6 h-6 text-primary" />
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            title={isCollapsed ? item.label : undefined}
            className={`w-full flex items-center gap-3 rounded-md px-3 py-2 text-base font-medium transition-colors ${
              activeTab === item.id
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            } ${isCollapsed ? 'justify-center' : ''}`}
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!isCollapsed && <span>{item.label}</span>}
          </button>
        ))}
      </nav>

      {/* Logout Button */}
      <div className="border-t mt-4">
        <button
          onClick={onLogout}
          title={isCollapsed ? 'Logout' : undefined}
          className={`w-full flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-all hover:bg-muted hover:text-foreground ${
            isCollapsed ? 'justify-center' : ''
          }`}
        >
          <LogOut className="h-5 w-5" />
          {!isCollapsed && <span>Logout</span>}
        </button>
      </div>
    </aside>
  )
}