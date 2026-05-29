import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { getHealth, listSessions, getUserStatus } from '../api'

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="card" style={{ flex: 1, minWidth: 130 }}>
      <div className="card-body" style={{ textAlign: 'center', padding: '20px 16px' }}>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: accent || 'var(--text)', lineHeight: 1 }}>
          {value}
        </div>
        <div style={{ fontWeight: 600, marginTop: 6, fontSize: '0.85rem' }}>{label}</div>
        {sub && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
      </div>
    </div>
  )
}

function timeAgo(ts) {
  if (!ts) return '—'
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function greeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

export default function Dashboard() {
  const { user, token } = useAuth()
  const isAdminOrOps = user?.role === 'admin' || user?.role === 'ops'

  const [health, setHealth] = useState(null)
  const [sessions, setSessions] = useState([])
  const [enrolledSelf, setEnrolledSelf] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetches = [getHealth()]
    if (isAdminOrOps) fetches.push(listSessions(token))
    else fetches.push(getUserStatus(user?.username))

    Promise.all(fetches)
      .then(([h, second]) => {
        setHealth(h)
        if (isAdminOrOps) setSessions(second ?? [])
        else setEnrolledSelf(second?.enrolled ?? false)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [token, isAdminOrOps, user?.username])

  const total = sessions.length
  const accepted = sessions.filter(s => s.status === 'accepted').length
  const locked = sessions.filter(s => s.is_locked).length
  const acceptRate = total > 0 ? Math.round((accepted / total) * 100) : null

  const recent = [...sessions]
    .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
    .slice(0, 5)

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">{greeting()}, {user?.username}</h1>
          <p className="page-subtitle">Vocalyx voice biometrics — system overview</p>
        </div>
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="empty-state-icon">⏳</div>
          <p>Loading…</p>
        </div>
      ) : (
        <>
          {/* Stat cards — admin / ops */}
          {isAdminOrOps && (
            <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
              <StatCard label="Enrolled users" value={health?.enrolled_users ?? '—'} />
              <StatCard label="Total sessions" value={total} />
              <StatCard
                label="Accept rate"
                value={acceptRate !== null ? `${acceptRate}%` : '—'}
                sub={total > 0 ? `${accepted} of ${total}` : 'no sessions yet'}
                accent={acceptRate !== null ? (acceptRate >= 70 ? 'var(--accept)' : 'var(--retry)') : undefined}
              />
              <StatCard
                label="Locked"
                value={locked}
                sub="sessions"
                accent={locked > 0 ? 'var(--reject)' : undefined}
              />
            </div>
          )}

          {/* Enrollment status — user role */}
          {user?.role === 'user' && (
            <div className="card" style={{ marginBottom: 20, maxWidth: 440 }}>
              <div className="card-body" style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{
                  fontSize: '1.5rem',
                  width: 40, height: 40,
                  borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: enrolledSelf ? 'var(--accept)' : 'var(--border)',
                  color: '#fff', flexShrink: 0,
                }}>
                  {enrolledSelf ? '✓' : '✕'}
                </div>
                <div>
                  <div style={{ fontWeight: 600 }}>Voice enrollment</div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    {enrolledSelf
                      ? 'You are enrolled and ready to authenticate'
                      : 'Not enrolled — visit Enroll to set up your voice profile'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Recent sessions — admin / ops */}
          {isAdminOrOps && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-section" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Recent sessions</span>
                <Link to="/sessions" style={{ fontSize: '0.82rem', color: 'var(--accent)', textDecoration: 'none' }}>
                  View all →
                </Link>
              </div>
              {recent.length === 0 ? (
                <div style={{ padding: '24px 20px', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  No sessions yet. Start an authentication to see activity here.
                </div>
              ) : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>User</th>
                        <th>Status</th>
                        <th>Attempts</th>
                        <th>When</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recent.map(s => (
                        <tr key={s.session_id}>
                          <td><strong>{s.user_id}</strong></td>
                          <td><span className={`badge badge-${s.status}`}>{s.status}</span></td>
                          <td>{s.total_attempts}</td>
                          <td style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>{timeAgo(s.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* System status */}
          {health && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-body" style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  System
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: 'var(--accept)', fontSize: '0.7rem' }}>●</span>
                  <span style={{ fontSize: '0.875rem' }}>API online</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: health.active_sessions > 0 ? 'var(--accent)' : 'var(--text-muted)', fontSize: '0.7rem' }}>●</span>
                  <span style={{ fontSize: '0.875rem' }}>{health.active_sessions} active session{health.active_sessions !== 1 ? 's' : ''}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ color: health.denoising ? 'var(--accept)' : 'var(--text-muted)', fontSize: '0.7rem' }}>●</span>
                  <span style={{ fontSize: '0.875rem' }}>Noise suppression {health.denoising ? 'on' : 'off'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Quick actions */}
          <div className="card">
            <div className="card-body">
              <div style={{ fontWeight: 600, marginBottom: 12, fontSize: '0.9rem' }}>Quick actions</div>
              <div className="flex gap-2">
                <Link to="/authenticate" className="btn btn-primary">Authenticate</Link>
                <Link to="/enroll" className="btn btn-secondary">Enroll voice</Link>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  )
}
