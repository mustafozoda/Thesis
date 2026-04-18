import React, { createContext, useContext, useState, useEffect } from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'

const DEFAULT_SERVER = 'http://192.168.137.1:8000'
const ServerContext = createContext()

export function ServerProvider({ children }) {
  const [server, setServer] = useState(DEFAULT_SERVER)

  useEffect(() => {
    AsyncStorage.getItem('serverUrl').then(saved => {
      if (saved) setServer(saved)
    })
  }, [])

  const saveServer = async (url) => {
    await AsyncStorage.setItem('serverUrl', url)
    setServer(url)
  }

  return (
    <ServerContext.Provider value={{ server, saveServer }}>
      {children}
    </ServerContext.Provider>
  )
}

export function useServer() {
  return useContext(ServerContext)
}