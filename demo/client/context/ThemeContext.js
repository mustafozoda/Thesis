import React, { createContext, useContext, useState } from 'react'

export const dark = {
  bg: '#0a0a0a',
  bgSecondary: '#141414',
  card: '#141414',
  cardAlt: '#1a1a1a',
  border: '#222',
  borderLight: '#2a2a2a',
  textPrimary: '#ffffff',
  textSec: '#888',
  textTertiary: '#555',
  textMuted: '#444',
  infoText: '#666',
  statusBar: 'light-content',
  switchTrackOff: '#333',
  hudBg: 'rgba(0,0,0,0.4)',
  analyzingBg: 'rgba(0,0,0,0.5)',
  analyzingCard: '#111',
  scanLine: 'rgba(255,255,255,0.08)',
  btnText: '#ffffff',
}

export const light = {
  bg: '#f0f0f0',
  bgSecondary: '#ffffff',
  card: '#ffffff',
  cardAlt: '#f7f7f7',
  border: '#e0e0e0',
  borderLight: '#ececec',
  textPrimary: '#0a0a0a',
  textSec: '#777',
  textTertiary: '#999',
  textMuted: '#bbb',
  infoText: '#888',
  statusBar: 'dark-content',
  switchTrackOff: '#ccc',
  hudBg: 'rgba(255,255,255,0.85)',
  analyzingBg: 'rgba(255,255,255,0.6)',
  analyzingCard: '#fff',
  scanLine: 'rgba(0,0,0,0.08)',
  btnText: '#ffffff',

}

const ThemeContext = createContext()

export function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(true)
  const toggle = () => setIsDark(d => !d)
  const theme = isDark ? dark : light
  return (
    <ThemeContext.Provider value={{ theme, isDark, toggle }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}