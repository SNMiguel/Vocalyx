import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const roleBadge = { admin: 'badge-admin', ops: 'badge-ops', user: 'badge-user' }

function SidebarLink({ to, icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
    >
      <span className="sidebar-link-icon">{icon}</span>
      {label}
    </NavLink>
  )
}

export default function Layout({ children }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="layout">
      <nav className="nav">
        <a className="nav-logo" href="/">
          <span className="nav-logo-icon">🎙️</span>
          Vocalyx
        </a>
        <div className="nav-spacer" />
        <div className="nav-user">
          <span className="nav-username">{user?.username}</span>
          <span className={`badge ${roleBadge[user?.role] ?? 'badge-user'}`}>{user?.role}</span>
          <button className="btn btn-secondary btn-sm" onClick={handleLogout}>
            Sign out
          </button>
        </div>
      </nav>

      <div className="body">
        <aside className="sidebar">
          <span className="sidebar-section">Voice Auth</span>
          <SidebarLink to="/authenticate" icon="🔐" label="Authenticate" />
          <SidebarLink to="/enroll" icon="🎤" label="Enroll" />

          {(user?.role === 'admin' || user?.role === 'ops') && (
            <>
              <span className="sidebar-section">Monitoring</span>
              <SidebarLink to="/sessions" icon="📋" label="Sessions" />
            </>
          )}

          {user?.role === 'admin' && (
            <>
              <span className="sidebar-section">Admin</span>
              <SidebarLink to="/users" icon="👤" label="Voice Users" />
            </>
          )}
        </aside>

        <main className="main">{children}</main>
      </div>
    </div>
  )
}
