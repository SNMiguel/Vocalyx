import { createContext, useContext, useState } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem('vocalyx_token'))
  const [user, setUser] = useState(() => {
    const s = localStorage.getItem('vocalyx_user')
    return s ? JSON.parse(s) : null
  })

  const login = (accessToken, userData) => {
    localStorage.setItem('vocalyx_token', accessToken)
    localStorage.setItem('vocalyx_user', JSON.stringify(userData))
    setToken(accessToken)
    setUser(userData)
  }

  const logout = () => {
    localStorage.removeItem('vocalyx_token')
    localStorage.removeItem('vocalyx_user')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ token, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
