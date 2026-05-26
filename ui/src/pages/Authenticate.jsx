import { useState, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { startSession, authenticate } from '../api'
import DecisionCard from '../components/DecisionCard'

export default function Authenticate() {
  const { user, token } = useAuth()
  const isAdmin = user?.role === 'admin'

  const [userId, setUserId] = useState(user?.username ?? '')
  const [sessionId, setSessionId] = useState(null)
  const [probeFile, setProbeFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState(1) // 1=start, 2=upload, 3=result
  const fileRef = useRef()

  const handleStartSession = async () => {
    if (!userId.trim()) return setError('User ID is required.')
    setError('')
    setLoading(true)
    try {
      const res = await startSession(userId.trim(), token)
      setSessionId(res.session_id)
      setStep(2)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAuthenticate = async () => {
    if (!probeFile) return setError('Select a probe audio file.')
    setError('')
    setLoading(true)
    try {
      const res = await authenticate(sessionId, probeFile, token)
      setResult(res)
      setStep(3)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setSessionId(null)
    setProbeFile(null)
    setResult(null)
    setError('')
    setStep(1)
  }

  const tryAgain = () => {
    setProbeFile(null)
    setResult(null)
    setError('')
    setStep(2)
  }

  const setFile = (f) => {
    if (f && (f.type.startsWith('audio/') || f.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i))) {
      setProbeFile(f)
    }
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Authenticate</h1>
          <p className="page-subtitle">Verify a speaker's identity using their enrolled voice profile.</p>
        </div>
      </div>

      <div style={{ maxWidth: 560 }}>
        {/* Step indicator */}
        <div className="step-indicator">
          <div className={`step ${step >= 1 ? (step > 1 ? 'done' : 'active') : ''}`}>
            <span className="step-num">{step > 1 ? '✓' : '1'}</span>
            <span>Start session</span>
          </div>
          <div className="step-sep" />
          <div className={`step ${step >= 2 ? (step > 2 ? 'done' : 'active') : ''}`}>
            <span className="step-num">{step > 2 ? '✓' : '2'}</span>
            <span>Upload probe</span>
          </div>
          <div className="step-sep" />
          <div className={`step ${step >= 3 ? 'active' : ''}`}>
            <span className="step-num">3</span>
            <span>Decision</span>
          </div>
        </div>

        {error && (
          <div className="alert alert-error" style={{ marginBottom: 16 }}>
            <span>⚠</span> {error}
          </div>
        )}

        {/* Step 1: Start session */}
        {step === 1 && (
          <div className="card">
            <div className="card-body">
              <div className="form-group">
                <label className="form-label" htmlFor="auth-uid">User ID to authenticate</label>
                <input
                  id="auth-uid"
                  className="form-input"
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
                  disabled={!isAdmin}
                  placeholder="e.g. alice"
                />
              </div>
              <button className="btn btn-primary btn-lg" onClick={handleStartSession} disabled={loading}>
                {loading ? 'Starting…' : 'Start session'}
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Upload probe */}
        {step === 2 && (
          <div className="card">
            <div className="card-body">
              <div className="alert alert-info" style={{ marginBottom: 16 }}>
                <span>ℹ</span> Session started for <strong>{userId}</strong>
                <span className="font-mono" style={{ marginLeft: 8, fontSize: '0.75rem', opacity: 0.7 }}>
                  {sessionId?.slice(0, 8)}…
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">Probe audio</label>
                <div
                  className={`drop-zone${dragging ? ' dragging' : ''}`}
                  onClick={() => fileRef.current.click()}
                  onDragOver={e => { e.preventDefault(); setDragging(true) }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={e => { e.preventDefault(); setDragging(false); setFile(e.dataTransfer.files[0]) }}
                  style={{ padding: '24px' }}
                >
                  {probeFile ? (
                    <>
                      <div className="drop-zone-icon">🎵</div>
                      <div className="drop-zone-text">{probeFile.name}</div>
                      <div className="drop-zone-sub">Click to change</div>
                    </>
                  ) : (
                    <>
                      <div className="drop-zone-icon">🎙</div>
                      <div className="drop-zone-text">Drop probe audio or click to browse</div>
                      <div className="drop-zone-sub">WAV · MP3 · FLAC</div>
                    </>
                  )}
                </div>
                <input
                  ref={fileRef}
                  type="file"
                  accept="audio/*"
                  style={{ display: 'none' }}
                  onChange={e => setFile(e.target.files[0])}
                />
              </div>

              <div className="flex gap-2">
                <button className="btn btn-primary btn-lg" onClick={handleAuthenticate} disabled={loading || !probeFile}>
                  {loading ? 'Verifying…' : 'Verify voice'}
                </button>
                <button className="btn btn-secondary" onClick={reset}>Cancel</button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Result */}
        {step === 3 && result && (
          <div>
            <DecisionCard result={result} />

            <div className="flex gap-2 mt-4">
              {(result.decision === 'retry' || result.decision === 'step_up') && (
                <button className="btn btn-primary" onClick={tryAgain}>Try again</button>
              )}
              <button className="btn btn-secondary" onClick={reset}>New session</button>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
