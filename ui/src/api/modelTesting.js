// API client for the admin Model Testing feature.
// Reuses BASE + ApiError from the main api module; adds multipart upload and
// a binary (Excel) download that the generic JSON `request` helper can't handle.
import { BASE, ApiError } from '../api'

async function jsonRequest(path, options = {}, token = null) {
  const headers = { ...(options.headers || {}) }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch {}
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return null
  return res.json()
}

function toQuery(filters = {}) {
  const params = new URLSearchParams()
  for (const [k, v] of Object.entries(filters)) {
    if (v !== '' && v != null) params.append(k, v)
  }
  const qs = params.toString()
  return qs ? `?${qs}` : ''
}

// Run the deepfake detector on one or more files with optional metadata.
export function runDeepfakeTest(files, metadata = {}, token) {
  const fd = new FormData()
  for (const f of files) fd.append('files', f)
  if (metadata.voice_model) fd.append('voice_model', metadata.voice_model)
  if (metadata.voice) fd.append('voice', metadata.voice)
  if (metadata.language) fd.append('language', metadata.language)
  if (metadata.notes) fd.append('notes', metadata.notes)
  return jsonRequest('/admin/test/deepfake', { method: 'POST', body: fd }, token)
}

// Paginated history. filters: { model_type, date_from, date_to, voice_model, tested_by, page, page_size }
export function getTestHistory(filters = {}, token) {
  return jsonRequest(`/admin/test/history${toQuery(filters)}`, {}, token)
}

// Trigger an Excel download for the given filters. filters: { model_type, date_from, date_to, voice_model }
export async function exportTestResults(filters = {}, token) {
  const headers = {}
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}/admin/test/export${toQuery(filters)}`, { headers })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch {}
    throw new ApiError(res.status, detail)
  }
  const blob = await res.blob()
  const cd = res.headers.get('Content-Disposition') || ''
  const m = cd.match(/filename="?([^"]+)"?/)
  const filename = m ? m[1] : 'vocalyx_model_tests.xlsx'
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export function deleteTestResult(testId, token) {
  return jsonRequest(`/admin/test/${encodeURIComponent(testId)}`, { method: 'DELETE' }, token)
}
