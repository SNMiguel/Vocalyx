import { useState, useRef, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import {
  runDeepfakeTest, getTestHistory, exportTestResults, deleteTestResult,
} from '../api/modelTesting'

const MAX_FILES = 10
const MAX_SIZE = 25 * 1024 * 1024  // 25 MB per file (matches backend)
const ACCEPT = '.wav,.mp3,.flac,.ogg,.m4a'
const PAGE_SIZE = 20

// Models selectable as tabs. Only the deepfake detector is wired up today;
// the rest are placeholders so the layout shows the intended roadmap.
const MODELS = [
  { id: 'deepfake_wav2vec2', label: 'Deepfake Detector', ready: true },
  { id: 'replay', label: 'Replay Detector', ready: false },
  { id: 'speaker', label: 'Speaker Verification', ready: false },
  { id: 'full_pipeline', label: 'Full Pipeline', ready: false },
]

function fmtBytes(b) {
  return b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`
}
function pct(x) {
  return x == null ? '—' : `${(x * 100).toFixed(2)}%`
}
function fmtTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return isNaN(d) ? iso : d.toLocaleString()
}
function LabelBadge({ label }) {
  if (!label) return <span className="text-muted">—</span>
  const cls = label === 'genuine' ? 'badge-accepted' : label === 'synthetic' ? 'badge-rejected' : 'badge-retry'
  return <span className={`badge ${cls}`}>{label}</span>
}

export default function ModelTesting() {
  const { token } = useAuth()
  const [model, setModel] = useState('deepfake_wav2vec2')

  // Upload + metadata
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const [meta, setMeta] = useState({ voice_model: '', voice: '', language: '', notes: '' })
  const inputRef = useRef()

  // Run state
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState([])     // current-run rows
  const [progress, setProgress] = useState({ completed: 0, total: 0 })
  const [error, setError] = useState('')

  // History
  const [history, setHistory] = useState([])
  const [historyTotal, setHistoryTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [filters, setFilters] = useState({ model_type: '', date_from: '', date_to: '', voice_model: '' })
  const [historyLoading, setHistoryLoading] = useState(false)
  const [sort, setSort] = useState({ key: 'tested_at', dir: 'desc' })

  const refreshHistory = useCallback(() => {
    setHistoryLoading(true)
    getTestHistory({ ...filters, page, page_size: PAGE_SIZE }, token)
      .then(data => { setHistory(data.results); setHistoryTotal(data.total) })
      .catch(err => setError(err.message))
      .finally(() => setHistoryLoading(false))
  }, [filters, page, token])

  useEffect(() => { refreshHistory() }, [refreshHistory])

  // ── file selection ─────────────────────────────────────────────────────────
  const addFiles = (incoming) => {
    setError('')
    const audio = Array.from(incoming).filter(
      f => f.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i)
    )
    const oversized = audio.filter(f => f.size > MAX_SIZE)
    if (oversized.length) {
      setError(`${oversized.length} file(s) exceed the 25 MB limit and were skipped.`)
    }
    const ok = audio.filter(f => f.size <= MAX_SIZE)
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      const merged = [...prev, ...ok.filter(f => !names.has(f.name))]
      if (merged.length > MAX_FILES) {
        setError(`Maximum ${MAX_FILES} files per run. Extra files were not added.`)
        return merged.slice(0, MAX_FILES)
      }
      return merged
    })
  }
  const removeFile = (i) => setFiles(prev => prev.filter((_, j) => j !== i))

  // ── run ────────────────────────────────────────────────────────────────────
  const runTests = async () => {
    if (files.length === 0) { setError('Add at least one audio file.'); return }
    setError('')
    setRunning(true)
    setProgress({ completed: 0, total: files.length })

    // Process one file per request so the table updates live as each completes.
    const rows = files.map(f => ({ filename: f.name, status: 'pending' }))
    setResults([...rows])

    for (let i = 0; i < files.length; i++) {
      rows[i] = { ...rows[i], status: 'processing' }
      setResults([...rows])
      try {
        const res = await runDeepfakeTest([files[i]], meta, token)
        const r = res.results[0]
        rows[i] = r.status === 'complete'
          ? { ...r, status: 'complete' }
          : { filename: files[i].name, status: 'failed', error: r.error }
      } catch (err) {
        rows[i] = { filename: files[i].name, status: 'failed', error: err.message }
      }
      setResults([...rows])
      setProgress({ completed: i + 1, total: files.length })
    }
    setRunning(false)
    setPage(1)
    refreshHistory()
  }

  // ── history actions ──────────────────────────────────────────────────────────
  const applyFilters = (patch) => { setPage(1); setFilters(prev => ({ ...prev, ...patch })) }

  const onExport = async () => {
    try {
      await exportTestResults(
        { model_type: filters.model_type, date_from: filters.date_from, date_to: filters.date_to, voice_model: filters.voice_model },
        token,
      )
    } catch (err) { setError(err.message) }
  }

  const onDelete = async (testId) => {
    try {
      await deleteTestResult(testId, token)
      refreshHistory()
    } catch (err) { setError(err.message) }
  }

  const sortedHistory = [...history].sort((a, b) => {
    const av = a[sort.key], bv = b[sort.key]
    if (av == null) return 1
    if (bv == null) return -1
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
    return sort.dir === 'asc' ? cmp : -cmp
  })
  const toggleSort = (key) =>
    setSort(s => s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' })
  const sortArrow = (key) => sort.key === key ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ''

  const totalPages = Math.max(1, Math.ceil(historyTotal / PAGE_SIZE))

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Model Testing</h1>
          <p className="page-subtitle">Run individual ML components in isolation — bypasses the full auth flow.</p>
        </div>
      </div>

      {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>⚠ {error}</div>}

      {/* Model selector */}
      <div className="tab-bar" style={{ marginBottom: 16 }}>
        {MODELS.map(m => (
          <button
            key={m.id}
            className={`tab-btn${model === m.id ? ' active' : ''}`}
            onClick={() => m.ready && setModel(m.id)}
            disabled={!m.ready}
            title={m.ready ? '' : 'Coming soon'}
          >
            {m.label}{!m.ready && ' (soon)'}
          </button>
        ))}
      </div>

      {/* Upload + run */}
      <div className="card mb-4">
        <div className="card-body">
          <div
            className={`drop-zone${dragging ? ' dragging' : ''}`}
            onClick={() => inputRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
          >
            <div className="drop-zone-icon">🧪</div>
            <div className="drop-zone-text">Drop audio files here or click to browse</div>
            <div className="drop-zone-sub">WAV · MP3 · FLAC · OGG · M4A · up to {MAX_FILES} files, 25 MB each</div>
          </div>
          <input ref={inputRef} type="file" accept={ACCEPT} multiple style={{ display: 'none' }}
                 onChange={e => addFiles(e.target.files)} />

          {files.length > 0 && (
            <div className="file-list">
              {files.map((f, i) => (
                <div key={i} className="file-item">
                  <span>🎵</span>
                  <span className="file-item-name">{f.name}</span>
                  <span className="file-item-size">{fmtBytes(f.size)}</span>
                  <button className="file-item-remove" onClick={() => removeFile(i)} title="Remove">×</button>
                </div>
              ))}
            </div>
          )}

          {/* Metadata */}
          <div className="flex gap-3 mt-4" style={{ flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: 1, minWidth: 160, marginBottom: 0 }}>
              <label className="form-label">Voice model <span className="text-muted">(optional)</span></label>
              <input className="form-input" placeholder="e.g. speech-2.8-hd"
                     value={meta.voice_model} onChange={e => setMeta({ ...meta, voice_model: e.target.value })} />
            </div>
            <div className="form-group" style={{ flex: 1, minWidth: 160, marginBottom: 0 }}>
              <label className="form-label">Voice <span className="text-muted">(optional)</span></label>
              <input className="form-input" placeholder="e.g. Calm Woman"
                     value={meta.voice} onChange={e => setMeta({ ...meta, voice: e.target.value })} />
            </div>
            <div className="form-group" style={{ flex: 1, minWidth: 140, marginBottom: 0 }}>
              <label className="form-label">Language <span className="text-muted">(optional)</span></label>
              <input className="form-input" placeholder="e.g. thai"
                     value={meta.language} onChange={e => setMeta({ ...meta, language: e.target.value })} />
            </div>
            <div className="form-group" style={{ flex: 2, minWidth: 200, marginBottom: 0 }}>
              <label className="form-label">Notes <span className="text-muted">(optional)</span></label>
              <input className="form-input" placeholder="freeform notes"
                     value={meta.notes} onChange={e => setMeta({ ...meta, notes: e.target.value })} />
            </div>
          </div>

          <button className="btn btn-primary btn-lg" style={{ marginTop: 16 }}
                  onClick={runTests} disabled={running || files.length === 0}>
            {running
              ? `Testing… ${progress.completed}/${progress.total}`
              : `Run Test${files.length > 1 ? ` (${files.length} files)` : ''}`}
          </button>
        </div>
      </div>

      {/* Current run results */}
      {results.length > 0 && (
        <div className="card mb-4">
          <div className="card-section">
            <strong>Results</strong>
            <span className="text-muted text-sm" style={{ marginLeft: 8 }}>
              {progress.completed} of {progress.total} completed
            </span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Filename</th><th>Label</th><th>Confidence</th>
                  <th>Deepfake Prob</th><th>Genuine Prob</th><th>Inference</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td className="file-item-name">{r.filename}</td>
                    <td><LabelBadge label={r.predicted_label} /></td>
                    <td className="font-mono">{pct(r.confidence)}</td>
                    <td className="font-mono">{pct(r.deepfake_prob)}</td>
                    <td className="font-mono">{pct(r.genuine_prob)}</td>
                    <td className="text-muted text-sm">{r.inference_ms != null ? `${r.inference_ms} ms` : '—'}</td>
                    <td>
                      {r.status === 'complete' && <span className="badge badge-accept">done</span>}
                      {r.status === 'processing' && <span className="badge badge-retry">processing…</span>}
                      {r.status === 'pending' && <span className="text-muted text-sm">queued</span>}
                      {r.status === 'failed' && (
                        <span className="badge badge-rejected" title={r.error}>failed</span>
                      )}
                      {r.status === 'failed' && r.error && (
                        <div className="text-sm" style={{ color: 'var(--reject)', marginTop: 2 }}>{r.error}</div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* History */}
      <div className="card">
        <div className="card-section flex items-center justify-between" style={{ flexWrap: 'wrap', gap: 8 }}>
          <strong>History</strong>
          <div className="flex gap-2 items-center" style={{ flexWrap: 'wrap' }}>
            <select className="form-input" style={{ width: 'auto' }}
                    value={filters.model_type} onChange={e => applyFilters({ model_type: e.target.value })}>
              <option value="">All models</option>
              <option value="deepfake_wav2vec2">Deepfake Detector</option>
            </select>
            <input className="form-input" type="date" style={{ width: 'auto' }} title="From"
                   value={filters.date_from} onChange={e => applyFilters({ date_from: e.target.value })} />
            <input className="form-input" type="date" style={{ width: 'auto' }} title="To"
                   value={filters.date_to} onChange={e => applyFilters({ date_to: e.target.value })} />
            <input className="form-input" style={{ width: 160 }} placeholder="Voice model…"
                   value={filters.voice_model} onChange={e => applyFilters({ voice_model: e.target.value })} />
            <button className="btn btn-secondary" onClick={refreshHistory}>↺</button>
            <button className="btn btn-primary" onClick={onExport} disabled={historyTotal === 0}>⬇ Export Excel</button>
          </div>
        </div>

        {historyLoading ? (
          <div className="empty-state"><div className="empty-state-icon">⏳</div><p>Loading…</p></div>
        ) : sortedHistory.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🧪</div>
            <div className="empty-state-title">No test results yet</div>
            <div className="empty-state-sub">Run a model test above and results will appear here.</div>
          </div>
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('tested_at')}>Tested At{sortArrow('tested_at')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('filename')}>Filename{sortArrow('filename')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('voice_model')}>Voice Model{sortArrow('voice_model')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('voice')}>Voice{sortArrow('voice')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('predicted_label')}>Label{sortArrow('predicted_label')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('confidence')}>Confidence{sortArrow('confidence')}</th>
                    <th>Deepfake</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('tested_by')}>By{sortArrow('tested_by')}</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {sortedHistory.map(r => (
                    <tr key={r.test_id}>
                      <td className="text-muted text-sm" style={{ whiteSpace: 'nowrap' }}>{fmtTime(r.tested_at)}</td>
                      <td className="file-item-name">{r.filename}</td>
                      <td>{r.voice_model || <span className="text-muted">—</span>}</td>
                      <td>{r.voice || <span className="text-muted">—</span>}</td>
                      <td><LabelBadge label={r.predicted_label} /></td>
                      <td className="font-mono">{pct(r.confidence)}</td>
                      <td className="font-mono">{pct(r.deepfake_prob)}</td>
                      <td className="text-sm">{r.tested_by}</td>
                      <td>
                        <button className="btn btn-danger btn-sm" onClick={() => onDelete(r.test_id)} title="Delete">×</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="card-section flex items-center justify-between">
              <span className="text-muted text-sm">{historyTotal} result(s)</span>
              <div className="flex gap-2 items-center">
                <button className="btn btn-secondary btn-sm" disabled={page <= 1}
                        onClick={() => setPage(p => Math.max(1, p - 1))}>← Prev</button>
                <span className="text-sm">Page {page} / {totalPages}</span>
                <button className="btn btn-secondary btn-sm" disabled={page >= totalPages}
                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}>Next →</button>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  )
}
