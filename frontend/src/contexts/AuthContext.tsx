import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import axios from 'axios'

interface User {
  email: string
  role: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, role?: string) => Promise<void>
  logout: () => void
  isAuthenticated: boolean
}

const AuthContext = createContext<AuthContextType | null>(null)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))

  useEffect(() => {
    if (token) {
      // Set default authorization header
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      
      // Try to get user info from token
      try {
        const tokenData = JSON.parse(atob(token.split('.')[1]))
        setUser({ email: tokenData.sub, role: tokenData.role || 'user' })
      } catch (error) {
        console.error('Invalid token:', error)
        logout()
      }
    }
  }, [token])

  const login = async (email: string, password: string) => {
    try {
      const response = await axios.post('/api/auth/login', { email, password })
      const { access_token } = response.data
      
      setToken(access_token)
      localStorage.setItem('token', access_token)
      
      // Decode token to get user info
      const tokenData = JSON.parse(atob(access_token.split('.')[1]))
      setUser({ email: tokenData.sub, role: tokenData.role || 'user' })
      
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
    } catch (error) {
      throw error
    }
  }

  const register = async (email: string, password: string, role: string = 'user') => {
    try {
      const response = await axios.post('/api/auth/register', { email, password, role })
      const { access_token } = response.data
      
      setToken(access_token)
      localStorage.setItem('token', access_token)
      
      // Decode token to get user info
      const tokenData = JSON.parse(atob(access_token.split('.')[1]))
      setUser({ email: tokenData.sub, role: tokenData.role || role })
      
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
    } catch (error) {
      throw error
    }
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    localStorage.removeItem('token')
    delete axios.defaults.headers.common['Authorization']
  }

  const value: AuthContextType = {
    user,
    token,
    login,
    register,
    logout,
    isAuthenticated: !!token && !!user,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
