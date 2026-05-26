const BASE = '/api'

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail)
    this.status = status
  }
}

async function request(path, options = {}, token = null) {
  const headers = { ...options.headers }
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(`${BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch {}
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return null
  return res.json()
}

// Auth
export const login = (username, password) =>
  request('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })

export const register = (username, password) =>
  request('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })

export const getMe = (token) => request('/auth/me', {}, token)

// App user management (admin only)
export const listAppUsers = (token) => request('/auth/users', {}, token)
export const updateUserRole = (username, role, token) =>
  request(`/auth/users/${encodeURIComponent(username)}`, {
    method: 'PATCH',
    body: JSON.stringify({ role }),
  }, token)
export const deleteAppUser = (username, token) =>
  request(`/auth/users/${encodeURIComponent(username)}`, { method: 'DELETE' }, token)

// Health
export const getHealth = () => request('/health')

// Voice users
export const getAllUsers = (token) => request('/users', {}, token)
export const getUserStatus = (userId) => request(`/users/${encodeURIComponent(userId)}`)
export const deleteUser = (userId, token) =>
  request(`/users/${encodeURIComponent(userId)}`, { method: 'DELETE' }, token)

// Enrollment
export const enrollUser = (userId, files, token) => {
  const fd = new FormData()
  fd.append('user_id', userId)
  for (const f of files) fd.append('files', f)
  return request('/enroll', { method: 'POST', body: fd }, token)
}

// Sessions
export const startSession = (userId, token) => {
  const fd = new FormData()
  fd.append('user_id', userId)
  return request('/sessions', { method: 'POST', body: fd }, token)
}

export const listSessions = (token) => request('/sessions', {}, token)

export const getSession = (sessionId, token) =>
  request(`/sessions/${encodeURIComponent(sessionId)}`, {}, token)

// Authenticate
export const authenticate = (sessionId, audioFile, token) => {
  const fd = new FormData()
  fd.append('session_id', sessionId)
  fd.append('file', audioFile)
  return request('/authenticate', { method: 'POST', body: fd }, token)
}
