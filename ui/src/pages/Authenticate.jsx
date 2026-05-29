import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { startSession, authenticate } from '../api'
import DecisionCard from '../components/DecisionCard'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

export default function Authenticate() {
  const { user, token } = useAuth()
  const isAdmin = user?.role === 'admin'

  const [userId, setUserId] = useState(user?.username ?? '')
  const [sessionId, setSessionId] = useState(null)
  const [challenge, setChallenge] = useState('')
  const [expiresIn, setExpiresIn] = useState(0)
  const [timeLeft, setTimeLeft] = useState(0)
  const [probeFile, setProbeFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState(1)

  // Refs so timer callbacks always see latest values
  const probeFileRef = useRef(null)
  const sessionIdRef = useRef(null)
  const recordingStopRef = useRef(null)
  const displayTimerRef = useRef(null)
  const expireTimerRef = useRef(null)

  useEffect(() => { probeFileRef.current = probeFile }, [probeFile])
  useEffect(() => { sessionIdRef.current = sessionId }, [sessionId])

  const { recording, elapsed, micError, micLevel, start, stop } = useAudioRecorder()

  // Keep stop ref fresh for the expiry callback
  useEffect(() => { recordingStopRef.current = stop }, [stop])

  const submitFile = useCallback(async (file, sid) => {
    if (!file || !sid) return
    setError('')
    setLoading(true)
    try {
      const res = await authenticate(sid, file, token)
      setResult(res)
    } catch (err) {
      // Always show a decision card — never leave user stuck on the recording screen
      setResult({
        decision: 'reject',
        speaker_score: 0,
        spoof_score: 0,
        explanation: err.message,
      })
    } finally {
      setLoading(false)
      setStep(3)
    }
  }, [token])

  // Called when countdown hits zero
  const handleExpired = useCallback(async () => {
    clearInterval(displayTimerRef.current)
    clearTimeout(expireTimerRef.current)

    let file = probeFileRef.current
    const sid = sessionIdRef.current

    // If mid-recording, stop and collect the clip
    if (recordingStopRef.current) {
      const res = await recordingStopRef.current()
      if (res) {
        const { wavBlob, duration } = res
        file = new File([wavBlob], `probe_${Date.now()}.wav`, { type: 'audio/wav' })
        file._previewUrl = URL.createObjectURL(wavBlob)
        file._duration = duration
        setProbeFile(file)
      }
    }

    if (file && sid) {
      // Auto-submit whatever was recorded
      await submitFile(file, sid)
    } else {
      setError('Session expired — no recording was made. Start a new session.')
      setSessionId(null)
      setChallenge('')
      setProbeFile(null)
      setStep(1)
    }
  }, [submitFile])

  // Start countdown when entering step 2
  useEffect(() => {
    if (step !== 2 || !expiresIn) return

    setTimeLeft(expiresIn)

    displayTimerRef.current = setInterval(() => {
      setTimeLeft(prev => Math.max(0, prev - 1))
    }, 1000)

    expireTimerRef.current = setTimeout(() => {
      clearInterval(displayTimerRef.current)
      handleExpired()
    }, expiresIn * 1000)

    return () => {
      clearInterval(displayTimerRef.current)
      clearTimeout(expireTimerRef.current)
    }
  }, [step, expiresIn, handleExpired])

  const handleStartSession = async () => {
    if (!userId.trim()) return setError('User ID is required.')
    setError('')
    setLoading(true)
    try {
      const res = await startSession(userId.trim(), token)
      setSessionId(res.session_id)
      setChallenge(res.challenge)
      setExpiresIn(res.expires_in ?? 10)
      setProbeFile(null)
      setResult(null)
      setStep(2)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAuthenticate = async () => {
    if (!probeFile) return setError('Record your voice first.')
    clearInterval(displayTimerRef.current)
    clearTimeout(expireTimerRef.current)
    await submitFile(probeFile, sessionId)
  }

  const reset = () => {
    clearInterval(displayTimerRef.current)
    clearTimeout(expireTimerRef.current)
    setSessionId(null)
    setChallenge('')
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

  const handleStopRecording = async () => {
    const res = await stop()
    if (!res) return
    const { wavBlob, duration } = res
    const file = new File([wavBlob], `probe_${Date.now()}.wav`, { type: 'audio/wav' })
    file._previewUrl = URL.createObjectURL(wavBlob)
    file._duration = duration
    setProbeFile(file)
  }

  const timerUrgent = timeLeft <= 5 && timeLeft > 0

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Authenticate</h1>
          <p className="page-subtitle">Verify a speaker's identity using their enrolled voice profile.</p>
        </div>
      </div>

      <div style={{ maxWidth: 560 }}>
        <div className="step-indicator">
          <div className={`step ${step >= 1 ? (step > 1 ? 'done' : 'active') : ''}`}>
            <span className="step-num">{step > 1 ? '✓' : '1'}</span>
            <span>Start session</span>
          </div>
          <div className="step-sep" />
          <div className={`step ${step >= 2 ? (step > 2 ? 'done' : 'active') : ''}`}>
            <span className="step-num">{step > 2 ? '✓' : '2'}</span>
            <span>Record voice</span>
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

        {/* Step 1 */}
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

        {/* Step 2 */}
        {step === 2 && (
          <div className="card">
            <div className="card-body">

              {/* Challenge phrase */}
              <div style={{ marginBottom: 16, padding: '16px', background: 'var(--accent)', borderRadius: 10 }}>
                <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.75)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>
                  Say this phrase aloud while recording
                </div>
                <div style={{ fontSize: '1.45rem', fontWeight: 700, letterSpacing: '0.03em', color: '#ffffff', lineHeight: 1.35 }}>
                  {challenge}
                </div>
              </div>

              {/* Countdown bar — hidden once submitted */}
              {!loading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                  <div style={{
                    fontSize: '0.85rem', fontWeight: 600, minWidth: 120,
                    color: timerUrgent ? 'var(--reject, #f38ba8)' : 'var(--text-muted)',
                    fontVariantNumeric: 'tabular-nums', transition: 'color 0.3s',
                  }}>
                    {timerUrgent ? '⚠ ' : ''}{timeLeft}s remaining
                  </div>
                  <div style={{ flex: 1, height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${(timeLeft / expiresIn) * 100}%`,
                      background: timerUrgent ? 'var(--reject, #f38ba8)' : 'var(--accent)',
                      transition: 'width 1s linear, background 0.3s',
                      borderRadius: 2,
                    }} />
                  </div>
                </div>
              )}

              {/* Recording controls — locked once loading or time is up */}
              {loading ? (
                <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)' }}>
                  <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>⏳</div>
                  <div style={{ fontWeight: 600 }}>Verifying your voice…</div>
                  <div style={{ fontSize: '0.85rem', marginTop: 4 }}>Please wait</div>
                </div>
              ) : (
                <div className="form-group">
                  <label className="form-label">Record yourself saying the phrase above</label>
                  {micError && <div className="alert alert-error" style={{ marginBottom: 8 }}>⚠ {micError}</div>}
                  <div className="record-zone" style={{ marginBottom: 12 }}>
                    <button
                      className={`record-btn${recording ? ' recording' : ''}`}
                      onClick={recording ? handleStopRecording : start}
                      disabled={timeLeft === 0}
                      title={recording ? 'Stop' : 'Start recording'}
                    >
                      {recording ? '⏹' : '●'}
                    </button>
                    {recording && (
                      <div className="mic-level-bar">
                        {Array.from({ length: 12 }).map((_, i) => (
                          <div key={i} className="mic-level-seg" style={{
                            opacity: micLevel > (i / 12) * 100 ? 1 : 0.15,
                            background: i < 8 ? 'var(--accept)' : 'var(--retry)',
                          }} />
                        ))}
                      </div>
                    )}
                    <div className="record-label">
                      {recording
                        ? <span className="record-live">Recording… {elapsed.toFixed(1)}s</span>
                        : probeFile
                          ? <span className="text-muted">Recording ready — submit or re-record</span>
                          : <span className="text-muted">Press ● then speak the phrase clearly</span>
                      }
                    </div>
                  </div>
                  {probeFile?._previewUrl && (
                    <audio src={probeFile._previewUrl} controls style={{ width: '100%', marginTop: 4 }} />
                  )}
                  <div className="flex gap-2" style={{ marginTop: 8 }}>
                    <button className="btn btn-primary btn-lg" onClick={handleAuthenticate} disabled={!probeFile || timeLeft === 0}>
                      Verify voice
                    </button>
                    <button className="btn btn-secondary" onClick={reset}>Cancel</button>
                  </div>
                </div>
              )}

            </div>
          </div>
        )}

        {/* Step 3 */}
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
