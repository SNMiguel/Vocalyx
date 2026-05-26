import { useState, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { enrollUser } from '../api'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

const MIN_SECONDS = 15  // guide shown to user (backend needs 10s after VAD)

const PROMPTS = [
  'Say your full name clearly.',
  'Tell us your age and where you\'re from.',
  'Describe what you do for work or study in a couple of sentences.',
  'Read this aloud: "The quick brown fox jumps over the lazy dog."',
]

function fmtBytes(b) {
  return b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`
}

function fmtTime(s) {
  if (s < 60) return `${s.toFixed(1)}s`
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

export default function Enroll() {
  const { user, token } = useAuth()
  const isAdmin = user?.role === 'admin'

  const [mode, setMode] = useState('record')
  const [userId, setUserId] = useState(isAdmin ? '' : user?.username ?? '')

  // Upload state
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  // Record state
  const [clips, setClips] = useState([])  // [{blob, duration, url, name}]
  const { recording, elapsed, micError, start, stop } = useAudioRecorder()

  // Shared
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  const totalDuration = clips.reduce((s, c) => s + c.duration, 0)
  const liveTotal = totalDuration + (recording ? elapsed : 0)
  const hasEnough = totalDuration >= MIN_SECONDS

  // ── upload helpers ───────────────────────────────────────────────────────────

  const addFiles = (incoming) => {
    const audio = Array.from(incoming).filter(
      f => f.type.startsWith('audio/') || f.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i)
    )
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...audio.filter(f => !names.has(f.name))]
    })
  }

  // ── recording helpers ────────────────────────────────────────────────────────

  const handleStopRecording = async () => {
    const result = await stop()
    if (!result) return
    const { wavBlob, duration } = result
    const url = URL.createObjectURL(wavBlob)
    setClips(prev => [...prev, { blob: wavBlob, duration, url, name: `clip_${prev.length + 1}.wav` }])
  }

  const removeClip = (i) => {
    URL.revokeObjectURL(clips[i].url)
    setClips(prev => prev.filter((_, j) => j !== i))
  }

  // ── submit ───────────────────────────────────────────────────────────────────

  const submit = async () => {
    if (!userId.trim()) return setStatus({ type: 'error', msg: 'User ID is required.' })

    let submitFiles
    if (mode === 'upload') {
      if (files.length === 0) return setStatus({ type: 'error', msg: 'Add at least one audio file.' })
      submitFiles = files
    } else {
      if (clips.length === 0) return setStatus({ type: 'error', msg: 'Record at least one clip.' })
      submitFiles = clips.map(c => new File([c.blob], c.name, { type: 'audio/wav' }))
    }

    setLoading(true)
    setStatus(null)
    try {
      const res = await enrollUser(userId.trim(), submitFiles, token)
      setStatus({ type: 'success', msg: res.message })
      setFiles([])
      clips.forEach(c => URL.revokeObjectURL(c.url))
      setClips([])
    } catch (err) {
      setStatus({ type: 'error', msg: err.message })
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = mode === 'upload' ? files.length > 0 : clips.length > 0

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Enroll Voice</h1>
          <p className="page-subtitle">Register a speaker profile using your voice.</p>
        </div>
      </div>

      <div className="card" style={{ maxWidth: 580 }}>
        <div className="card-body">
          {status && (
            <div className={`alert alert-${status.type}`}>
              {status.type === 'success' ? '✓' : '⚠'} {status.msg}
            </div>
          )}

          {/* User ID */}
          <div className="form-group">
            <label className="form-label" htmlFor="uid">User ID</label>
            <input
              id="uid"
              className="form-input"
              value={userId}
              onChange={e => setUserId(e.target.value)}
              disabled={!isAdmin}
              placeholder="e.g. alice"
            />
            {!isAdmin && (
              <p className="text-sm text-muted" style={{ marginTop: 4 }}>
                You can only enroll your own account.
              </p>
            )}
          </div>

          {/* Mode tabs */}
          <div className="tab-bar">
            <button className={`tab-btn${mode === 'record' ? ' active' : ''}`} onClick={() => setMode('record')}>
              🎙 Record
            </button>
            <button className={`tab-btn${mode === 'upload' ? ' active' : ''}`} onClick={() => setMode('upload')}>
              📁 Upload files
            </button>
          </div>

          {/* ── Record mode ────────────────────────────────────────────────── */}
          {mode === 'record' && (
            <div>
              <div className="prompt-card">
                <div className="prompt-card-title">What to say</div>
                <ol className="prompt-list">
                  {PROMPTS.map((p, i) => <li key={i}>{p}</li>)}
                </ol>
                <p className="text-sm text-muted" style={{ marginTop: 8 }}>
                  Read through all four prompts. You can record in multiple clips and remove any you want to redo.
                </p>
              </div>

              {micError && (
                <div className="alert alert-error" style={{ marginBottom: 12 }}>⚠ {micError}</div>
              )}

              {/* Record button */}
              <div className="record-zone">
                <button
                  className={`record-btn${recording ? ' recording' : ''}`}
                  onClick={recording ? handleStopRecording : start}
                  title={recording ? 'Stop recording' : 'Start recording'}
                >
                  {recording ? '⏹' : '●'}
                </button>
                <div className="record-label">
                  {recording
                    ? <span className="record-live">Recording… {fmtTime(elapsed)}</span>
                    : <span className="text-muted">{clips.length === 0 ? 'Press to start' : 'Record another clip'}</span>
                  }
                </div>
              </div>

              {/* Duration progress */}
              {(clips.length > 0 || recording) && (
                <div className="duration-wrap">
                  <div className="duration-header">
                    <span className="text-sm">Voice data captured</span>
                    <span className={`text-sm duration-value${hasEnough ? ' enough' : ''}`}>
                      {fmtTime(liveTotal)} / {fmtTime(MIN_SECONDS)} min
                    </span>
                  </div>
                  <div className="duration-bar">
                    <div
                      className="duration-bar-fill"
                      style={{
                        width: `${Math.min((liveTotal / MIN_SECONDS) * 100, 100)}%`,
                        background: hasEnough ? 'var(--accept)' : 'var(--accent)',
                        transition: 'width 0.1s, background 0.3s',
                      }}
                    />
                  </div>
                  {hasEnough && (
                    <p className="text-sm" style={{ color: 'var(--accept)', marginTop: 4 }}>
                      ✓ Enough voice data captured for a reliable profile.
                    </p>
                  )}
                  {!hasEnough && !recording && clips.length > 0 && (
                    <p className="text-sm text-muted" style={{ marginTop: 4 }}>
                      Record a bit more to improve accuracy. {fmtTime(MIN_SECONDS - totalDuration)} remaining.
                    </p>
                  )}
                </div>
              )}

              {/* Clips */}
              {clips.length > 0 && (
                <div className="file-list">
                  {clips.map((c, i) => (
                    <div key={i} className="file-item" style={{ alignItems: 'center' }}>
                      <audio src={c.url} controls style={{ height: 28, flex: 1, minWidth: 0 }} />
                      <span className="file-item-size">{fmtTime(c.duration)}</span>
                      <button className="file-item-remove" onClick={() => removeClip(i)} title="Remove clip">×</button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── Upload mode ────────────────────────────────────────────────── */}
          {mode === 'upload' && (
            <div>
              <div
                className={`drop-zone${dragging ? ' dragging' : ''}`}
                onClick={() => inputRef.current.click()}
                onDragOver={e => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
              >
                <div className="drop-zone-icon">🎙</div>
                <div className="drop-zone-text">Drop audio files here or click to browse</div>
                <div className="drop-zone-sub">WAV · MP3 · FLAC · OGG  (16kHz recommended, ≥ 10s of speech)</div>
              </div>
              <input ref={inputRef} type="file" accept="audio/*" multiple style={{ display: 'none' }} onChange={e => addFiles(e.target.files)} />

              {files.length > 0 && (
                <div className="file-list">
                  {files.map((f, i) => (
                    <div key={i} className="file-item">
                      <span>🎵</span>
                      <span className="file-item-name">{f.name}</span>
                      <span className="file-item-size">{fmtBytes(f.size)}</span>
                      <button className="file-item-remove" onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))} title="Remove">×</button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <button
            className="btn btn-primary btn-lg"
            style={{ marginTop: 8 }}
            onClick={submit}
            disabled={loading || !canSubmit}
          >
            {loading ? 'Enrolling…' : `Enroll${mode === 'upload' && files.length > 1 ? ` (${files.length} files)` : mode === 'record' && clips.length > 1 ? ` (${clips.length} clips)` : ''}`}
          </button>
        </div>
      </div>
    </>
  )
}
