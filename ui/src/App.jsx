import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import Layout from './components/Layout'
import Login from './pages/Login'
import Enroll from './pages/Enroll'
import Authenticate from './pages/Authenticate'
import Sessions from './pages/Sessions'
import Users from './pages/Users'

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/" element={<Navigate to="/authenticate" replace />} />
                    <Route path="/authenticate" element={<Authenticate />} />
                    <Route path="/enroll" element={<Enroll />} />
                    <Route
                      path="/sessions"
                      element={
                        <ProtectedRoute roles={['admin', 'ops']}>
                          <Sessions />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/users"
                      element={
                        <ProtectedRoute roles={['admin']}>
                          <Users />
                        </ProtectedRoute>
                      }
                    />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
