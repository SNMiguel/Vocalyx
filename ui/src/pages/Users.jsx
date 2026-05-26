import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { getAllUsers, deleteUser } from '../api'

export default function Users() {
  const { token } = useAuth()
  const navigate = useNavigate()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleting, setDeleting] = useState(null)

  const load = () => {
    setLoading(true)
    getAllUsers(token)
      .then(data => setUsers(data.users ?? []))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [token])

  const handleDelete = async (userId) => {
    if (!confirm(`Delete voice enrollment for "${userId}"? This cannot be undone.`)) return
    setDeleting(userId)
    try {
      await deleteUser(userId, token)
      setUsers(prev => prev.filter(u => u !== userId))
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Voice Users</h1>
          <p className="page-subtitle">All speaker profiles enrolled in the system.</p>
        </div>
        <div className="flex gap-2">
          <button className="btn btn-secondary" onClick={load}>↺ Refresh</button>
          <button className="btn btn-primary" onClick={() => navigate('/enroll')}>+ Enroll user</button>
        </div>
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
            <div className="empty-state-icon">👤</div>
            <div className="empty-state-title">No voice users enrolled</div>
            <div className="empty-state-sub">
              <button className="btn btn-primary" style={{ marginTop: 12 }} onClick={() => navigate('/enroll')}>
                Enroll first user
              </button>
            </div>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>User ID</th>
                  <th>Status</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map(uid => (
                  <tr key={uid}>
                    <td><strong>{uid}</strong></td>
                    <td><span className="badge badge-accept">Enrolled</span></td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="flex gap-2" style={{ justifyContent: 'flex-end' }}>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => navigate(`/authenticate?uid=${encodeURIComponent(uid)}`)}
                        >
                          Authenticate
                        </button>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => handleDelete(uid)}
                          disabled={deleting === uid}
                        >
                          {deleting === uid ? 'Deleting…' : 'Delete'}
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
