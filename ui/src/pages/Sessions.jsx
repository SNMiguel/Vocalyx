import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { listSessions } from '../api'

const decisionBadge = (d) => `badge badge-${d}`
const statusBadge = (s) => `badge badge-${s}`

function AttemptList({ attempts }) {
  if (!attempts?.length) return <p className="text-muted text-sm" style={{ padding: '8px 16px 8px 40px' }}>No attempts recorded.</p>
  return (
    <div className="attempt-list">
      {attempts.map(a => (
        <div key={a.attempt} className="attempt-item">
          <span className="attempt-num">#{a.attempt}</span>
          <span className={decisionBadge(a.decision)}>{a.decision}</span>
          <span className="text-sm">SV: {(a.speaker_score * 100).toFixed(0)}%</span>
          <span className="text-sm">Spoof: {(a.spoof_score * 100).toFixed(0)}%</span>
          <span className="attempt-expl">{a.explanation}</span>
        </div>
      ))}
    </div>
  )
}

export default function Sessions() {
  const { token } = useAuth()
  const [sessions, setSessions] = useState([])
  const [expanded, setExpanded] = useState(new Set())
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    listSessions(token)
      .then(setSessions)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [token])

  const toggle = (id) => setExpanded(prev => {
    const next = new Set(prev)
    next.has(id) ? next.delete(id) : next.add(id)
    return next
  })

  const filtered = sessions.filter(s =>
    !filter || s.user_id.toLowerCase().includes(filter.toLowerCase())
  )

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Sessions</h1>
          <p className="page-subtitle">Auth attempt history for all enrolled users.</p>
        </div>
        <button className="btn btn-secondary" onClick={() => {
          setLoading(true)
          listSessions(token).then(setSessions).finally(() => setLoading(false))
        }}>
          ↺ Refresh
        </button>
      </div>

      {error && <div className="alert alert-error">⚠ {error}</div>}

      <div className="card">
        <div className="card-section">
          <input
            className="form-input"
            placeholder="Filter by user ID…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            style={{ maxWidth: 300 }}
          />
        </div>

        {loading ? (
          <div className="empty-state">
            <div className="empty-state-icon">⏳</div>
            <p>Loading sessions…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div className="empty-state-title">No sessions yet</div>
            <div className="empty-state-sub">Sessions appear here after authentication attempts are made.</div>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th />
                  <th>Session ID</th>
                  <th>User</th>
                  <th>Status</th>
                  <th>Attempts</th>
                  <th>Retries</th>
                  <th>Locked</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(s => (
                  <>
                    <tr key={s.session_id}>
                      <td>
                        <button className="expand-btn" onClick={() => toggle(s.session_id)}>
                          {expanded.has(s.session_id) ? '▾' : '▸'}
                        </button>
                      </td>
                      <td><span className="font-mono truncate">{s.session_id}</span></td>
                      <td><strong>{s.user_id}</strong></td>
                      <td><span className={statusBadge(s.status)}>{s.status}</span></td>
                      <td>{s.total_attempts}</td>
                      <td>{s.retry_count}</td>
                      <td>{s.is_locked ? '🔒 Yes' : '—'}</td>
                    </tr>
                    {expanded.has(s.session_id) && (
                      <tr key={`${s.session_id}-exp`} className="attempt-row">
                        <td colSpan={7} style={{ padding: 0 }}>
                          <AttemptList attempts={s.attempts} />
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
