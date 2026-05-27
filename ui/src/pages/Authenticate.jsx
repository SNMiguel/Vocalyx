import { useState, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { startSession, authenticate } from '../api'
import DecisionCard from '../components/DecisionCard'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

export default function Authenticate() {
  const { user, token } = useAuth()
  const isAdmin = user?.role === 'admin'

  const [userId, setUserId] = useState(user?.username ?? '')
  const [sessionId, setSessionId] = useState(null)
  const [probeFile, setProbeFile] = useState(null)
  const [probeMode, setProbeMode] = useState('record') // 'record' | 'upload'
  const [dragging, setDragging] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState(1) // 1=start, 2=probe, 3=result
  const fileRef = useRef()
  const { recording, elapsed, micError, micLevel, start, stop } = useAudioRecorder()

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

  const handleStopRecording = async () => {
    const result = await stop()
    if (!result) return
    const { wavBlob, duration } = result
    const file = new File([wavBlob], `probe_${Date.now()}.wav`, { type: 'audio/wav' })
    file._previewUrl = URL.createObjectURL(wavBlob)
    file._duration = duration
    setProbeFile(file)
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
            <span>Probe audio</span>
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

        {/* Step 2: Probe audio */}
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

                {/* Mode tabs */}
                <div className="tab-bar" style={{ marginBottom: 12 }}>
                  <button className={`tab-btn${probeMode === 'record' ? ' active' : ''}`} onClick={() => { setProbeMode('record'); setProbeFile(null) }}>
                    🎙 Record
                  </button>
                  <button className={`tab-btn${probeMode === 'upload' ? ' active' : ''}`} onClick={() => { setProbeMode('upload'); setProbeFile(null) }}>
                    📁 Upload
                  </button>
                </div>

                {probeMode === 'record' ? (
                  <div>
                    {micError && <div className="alert alert-error" style={{ marginBottom: 8 }}>⚠ {micError}</div>}
                    <div className="record-zone" style={{ marginBottom: 12 }}>
                      <button
                        className={`record-btn${recording ? ' recording' : ''}`}
                        onClick={recording ? handleStopRecording : start}
                        title={recording ? 'Stop' : 'Start recording'}
                      >
                        {recording ? '⏹' : '●'}
                      </button>
                      {recording && (
                        <div className="mic-level-bar">
                          {Array.from({ length: 12 }).map((_, i) => (
                            <div
                              key={i}
                              className="mic-level-seg"
                              style={{
                                opacity: micLevel > (i / 12) * 100 ? 1 : 0.15,
                                background: i < 8 ? 'var(--accept)' : 'var(--retry)',
                              }}
                            />
                          ))}
                        </div>
                      )}
                      <div className="record-label">
                        {recording
                          ? <span className="record-live">Recording… {elapsed.toFixed(1)}s</span>
                          : probeFile
                            ? <span className="text-muted">Recording ready — re-record to replace</span>
                            : <span className="text-muted">Say a few sentences in your natural voice</span>
                        }
                      </div>
                    </div>
                    {probeFile?._previewUrl && (
                      <audio src={probeFile._previewUrl} controls style={{ width: '100%', marginTop: 4 }} />
                    )}
                  </div>
                ) : (
                  <div>
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
                          <div className="drop-zone-sub">WAV · FLAC · OGG · M4A · MP3</div>
                        </>
                      )}
                    </div>
                    <input ref={fileRef} type="file" accept="audio/*" style={{ display: 'none' }} onChange={e => setFile(e.target.files[0])} />
                  </div>
                )}
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
