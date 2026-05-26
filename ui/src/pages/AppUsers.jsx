import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { listAppUsers, updateUserRole, deleteAppUser } from '../api'

const ROLES = ['user', 'ops', 'admin']

const roleBadgeClass = { admin: 'badge-admin', ops: 'badge-ops', user: 'badge-user' }

export default function AppUsers() {
  const { token, user: self } = useAuth()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(null)

  const load = () => {
    setLoading(true)
    listAppUsers(token)
      .then(data => setUsers(data.users ?? []))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [token])

  const handleRoleChange = async (username, role) => {
    setBusy(username)
    try {
      await updateUserRole(username, role, token)
      setUsers(prev => prev.map(u => u.username === username ? { ...u, role } : u))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const handleDelete = async (username) => {
    if (!confirm(`Delete account "${username}"? This cannot be undone.`)) return
    setBusy(username)
    try {
      await deleteAppUser(username, token)
      setUsers(prev => prev.filter(u => u.username !== username))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">App Users</h1>
          <p className="page-subtitle">Manage dashboard accounts and role assignments.</p>
        </div>
        <button className="btn btn-secondary" onClick={load}>↺ Refresh</button>
      </div>

      {error && <div className="alert alert-error">⚠ {error}</div>}

      <div className="card">
        {loading ? (
          <div className="empty-state">
            <div className="empty-state-icon">⏳</div>
            <p>Loading users…</p>
          </div>
        ) : users.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🔑</div>
            <div className="empty-state-title">No app users found</div>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Role</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.username}>
                    <td>
                      <strong>{u.username}</strong>
                      {u.username === self?.username && (
                        <span className="text-muted text-sm" style={{ marginLeft: 8 }}>(you)</span>
                      )}
                    </td>
                    <td>
                      <span className={`badge ${roleBadgeClass[u.role] ?? 'badge-user'}`}>
                        {u.role}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="flex gap-2" style={{ justifyContent: 'flex-end' }}>
                        <select
                          className="form-input"
                          style={{ width: 'auto', padding: '4px 8px', fontSize: 13 }}
                          value={u.role}
                          disabled={busy === u.username || u.username === self?.username}
                          onChange={e => handleRoleChange(u.username, e.target.value)}
                        >
                          {ROLES.map(r => (
                            <option key={r} value={r}>{r}</option>
                          ))}
                        </select>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => handleDelete(u.username)}
                          disabled={busy === u.username || u.username === self?.username}
                        >
                          {busy === u.username ? '…' : 'Delete'}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
