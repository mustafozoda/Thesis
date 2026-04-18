import React, { createContext, useContext, useState, useEffect } from 'react'
import AsyncStorage from '@react-native-async-storage/async-storage'

const HISTORY_KEY = 'scan_history'
const MAX_ENTRIES = 50

const ScanHistoryContext = createContext()

export function ScanHistoryProvider({ children }) {
  const [history, setHistory] = useState([])

  useEffect(() => {
    AsyncStorage.getItem(HISTORY_KEY).then(raw => {
      if (raw) {
        try { setHistory(JSON.parse(raw)) } catch (_) { }
      }
    })
  }, [])

  const addScan = async (entry) => {
    const newEntry = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      ...entry,
    }
    setHistory(prev => {
      const updated = [newEntry, ...prev].slice(0, MAX_ENTRIES)
      AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
    return newEntry.id
  }

  const clearHistory = async () => {
    setHistory([])
    await AsyncStorage.removeItem(HISTORY_KEY)
  }

  const removeEntry = async (id) => {
    setHistory(prev => {
      const updated = prev.filter(e => e.id !== id)
      AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
      return updated
    })
  }

  return (
    <ScanHistoryContext.Provider value={{ history, addScan, clearHistory, removeEntry }}>
      {children}
    </ScanHistoryContext.Provider>
  )
}

export function useScanHistory() {
  return useContext(ScanHistoryContext)
}