import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { getAuditLog } from '../api'

const ACTION_BADGE = {
  enroll:         'badge-accepted',
  delete_voice:   'badge-rejected',
  delete_account: 'badge-rejected',
  change_role:    'badge-step_up',
}

function timeAgo(ts) {
  if (!ts) return '—'
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(ts * 1000).toLocaleDateString()
}

export default function AuditLog() {
  const { token } = useAuth()
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')

  const refresh = useCallback(() => {
    setLoading(true)
    getAuditLog(token)
      .then(data => { setEntries(data); setError('') })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [token])

  useEffect(() => { refresh() }, [refresh])

  const filtered = entries.filter(e =>
    !filter ||
    e.actor.toLowerCase().includes(filter.toLowerCase()) ||
    e.action.toLowerCase().includes(filter.toLowerCase()) ||
    (e.target ?? '').toLowerCase().includes(filter.toLowerCase())
  )

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Audit Log</h1>
          <p className="page-subtitle">Admin actions — enroll, delete, role changes.</p>
        </div>
        <button className="btn btn-secondary" onClick={refresh}>↺ Refresh</button>
      </div>

      {error && <div className="alert alert-error">⚠ {error}</div>}

      <div className="card">
        <div className="card-section">
          <input
            className="form-input"
            placeholder="Filter by actor, action, or target…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            style={{ maxWidth: 360 }}
          />
        </div>

        {loading ? (
          <div className="empty-state">
            <div className="empty-state-icon">⏳</div>
            <p>Loading audit log…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div className="empty-state-title">No entries yet</div>
            <div className="empty-state-sub">Admin actions will appear here.</div>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Actor</th>
                  <th>Action</th>
                  <th>Target</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(e => (
                  <tr key={e.id}>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem', whiteSpace: 'nowrap' }}>
                      {timeAgo(e.timestamp)}
                    </td>
                    <td><strong>{e.actor}</strong></td>
                    <td>
                      <span className={`badge ${ACTION_BADGE[e.action] ?? 'badge-retry'}`}>
                        {e.action}
                      </span>
                    </td>
                    <td>{e.target ?? '—'}</td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{e.details ?? '—'}</td>
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
