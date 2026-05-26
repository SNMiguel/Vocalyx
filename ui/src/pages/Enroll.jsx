import { useState, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { enrollUser } from '../api'

function formatBytes(b) {
  return b < 1024 ? `${b} B` : b < 1048576 ? `${(b/1024).toFixed(1)} KB` : `${(b/1048576).toFixed(1)} MB`
}

export default function Enroll() {
  const { user, token } = useAuth()
  const isAdmin = user?.role === 'admin'

  const [userId, setUserId] = useState(isAdmin ? '' : user?.username ?? '')
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState(null) // { type: 'success'|'error', msg }
  const [loading, setLoading] = useState(false)
  const inputRef = useRef()

  const addFiles = (incoming) => {
    const audio = Array.from(incoming).filter(f => f.type.startsWith('audio/') || f.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i))
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...audio.filter(f => !names.has(f.name))]
    })
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }

  const submit = async () => {
    if (!userId.trim()) return setStatus({ type: 'error', msg: 'User ID is required.' })
    if (files.length === 0) return setStatus({ type: 'error', msg: 'Add at least one audio file.' })
    setLoading(true)
    setStatus(null)
    try {
      const res = await enrollUser(userId.trim(), files, token)
      setStatus({ type: 'success', msg: res.message })
      setFiles([])
    } catch (err) {
      setStatus({ type: 'error', msg: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Enroll Voice</h1>
          <p className="page-subtitle">Upload audio samples to register a speaker profile.</p>
        </div>
      </div>

      <div className="card" style={{ maxWidth: 560 }}>
        <div className="card-body">
          {status && (
            <div className={`alert alert-${status.type}`}>
              {status.type === 'success' ? '✓' : '⚠'} {status.msg}
            </div>
          )}

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

          <div className="form-group">
            <label className="form-label">Audio samples</label>
            <div
              className={`drop-zone${dragging ? ' dragging' : ''}`}
              onClick={() => inputRef.current.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div className="drop-zone-icon">🎙</div>
              <div className="drop-zone-text">Drop audio files here or click to browse</div>
              <div className="drop-zone-sub">WAV · MP3 · FLAC · OGG  (16kHz recommended)</div>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="audio/*"
              multiple
              style={{ display: 'none' }}
              onChange={e => addFiles(e.target.files)}
            />

            {files.length > 0 && (
              <div className="file-list">
                {files.map((f, i) => (
                  <div key={i} className="file-item">
                    <span>🎵</span>
                    <span className="file-item-name">{f.name}</span>
                    <span className="file-item-size">{formatBytes(f.size)}</span>
                    <button
                      className="file-item-remove"
                      onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))}
                      title="Remove"
                    >×</button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button className="btn btn-primary btn-lg" onClick={submit} disabled={loading}>
            {loading ? 'Enrolling…' : `Enroll${files.length > 1 ? ` (${files.length} files)` : ''}`}
          </button>
        </div>
      </div>
    </>
  )
}
