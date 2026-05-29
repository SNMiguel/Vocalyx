import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { listSessions } from '../api'

const REFRESH_INTERVAL = 30 // seconds

function timeAgo(ts) {
  if (!ts) return '—'
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function AttemptList({ attempts, challenge }) {
  return (
    <div style={{ padding: '8px 16px 12px 40px' }}>
      {challenge && (
        <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Challenge</span>
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--accent)' }}>{challenge}</span>
        </div>
      )}
      {!attempts?.length
        ? <p className="text-muted text-sm">No attempts recorded.</p>
        : <div className="attempt-list">
            {attempts.map(a => (
              <div key={a.attempt} className="attempt-item">
                <span className="attempt-num">#{a.attempt}</span>
                <span className={`badge badge-${a.decision}`}>{a.decision}</span>
                <span className="text-sm">SV: {(a.speaker_score * 100).toFixed(0)}%</span>
                <span className="text-sm">Spoof: {(a.spoof_score * 100).toFixed(0)}%</span>
                <span className="attempt-expl">{a.explanation}</span>
              </div>
            ))}
          </div>
      }
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
  const [lastUpdated, setLastUpdated] = useState(null)
  const [secondsSince, setSecondsSince] = useState(0)
  const intervalRef = useRef(null)
  const tickRef = useRef(null)

  const refresh = useCallback((silent = false) => {
    if (!silent) setLoading(true)
    listSessions(token)
      .then(data => {
        setSessions(data)
        setLastUpdated(Date.now())
        setSecondsSince(0)
        setError('')
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [token])

  // Initial load + auto-refresh every 30s
  useEffect(() => {
    refresh()
    intervalRef.current = setInterval(() => refresh(true), REFRESH_INTERVAL * 1000)
    return () => clearInterval(intervalRef.current)
  }, [refresh])

  // Tick counter for "last updated Xs ago"
  useEffect(() => {
    tickRef.current = setInterval(() => {
      setSecondsSince(prev => prev + 1)
    }, 1000)
    return () => clearInterval(tickRef.current)
  }, [])

  const toggle = (id) => setExpanded(prev => {
    const next = new Set(prev)
    next.has(id) ? next.delete(id) : next.add(id)
    return next
  })

  const filtered = [...sessions]
    .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
    .filter(s => !filter || s.user_id.toLowerCase().includes(filter.toLowerCase()))

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Sessions</h1>
          <p className="page-subtitle">Auth attempt history for all enrolled users.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {lastUpdated && (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Updated {secondsSince}s ago · auto-refreshes every {REFRESH_INTERVAL}s
            </span>
          )}
          <button className="btn btn-secondary" onClick={() => refresh()}>↺ Refresh</button>
        </div>
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
                  <th>User</th>
                  <th>Status</th>
                  <th>Attempts</th>
                  <th>Retries</th>
                  <th>Locked</th>
                  <th>Started</th>
                  <th>Completed</th>
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
                      <td><strong>{s.user_id}</strong></td>
                      <td><span className={`badge badge-${s.status}`}>{s.status}</span></td>
                      <td>{s.total_attempts}</td>
                      <td>{s.retry_count}</td>
                      <td>{s.is_locked ? '🔒 Yes' : '—'}</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{timeAgo(s.created_at)}</td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{s.completed_at ? timeAgo(s.completed_at) : <span style={{ color: 'var(--accent)' }}>active</span>}</td>
                    </tr>
                    {expanded.has(s.session_id) && (
                      <tr key={`${s.session_id}-exp`} className="attempt-row">
                        <td colSpan={8} style={{ padding: 0 }}>
                          <AttemptList attempts={s.attempts} challenge={s.challenge} />
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
