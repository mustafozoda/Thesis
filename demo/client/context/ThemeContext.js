import React, { createContext, useContext, useState } from 'react'

export const dark = {
  // Backgrounds
  bg: '#080808',
  bgSecondary: '#101010',
  card: '#141414',
  cardAlt: '#1c1c1c',
  cardElevated: '#1a1a1a',

  // Borders
  border: '#1e1e1e',
  borderLight: '#242424',
  borderMid: '#2a2a2a',

  // Text
  textPrimary: '#f0f0f0',
  textSec: '#888',
  textTertiary: '#555',
  textMuted: '#383838',
  infoText: '#5a5a5a',

  // Status bar
  statusBar: 'light-content',

  // UI elements
  switchTrackOff: '#2a2a2a',
  hudBg: 'rgba(8,8,8,0.75)',
  analyzingBg: 'rgba(0,0,0,0.65)',
  analyzingCard: '#111',
  scanLine: 'rgba(255,255,255,0.06)',

  // Buttons
  btnText: '#ffffff',
  btnSecBg: '#1c1c1c',

  // Accent
  accent: '#1D9E75',
  accentDim: 'rgba(29,158,117,0.15)',
  accentBorder: 'rgba(29,158,117,0.35)',

  // Step colors
  step1: '#378ADD',
  step1Dim: 'rgba(55,138,221,0.15)',
  step2: '#1D9E75',
  step2Dim: 'rgba(29,158,117,0.15)',
  step3: '#EF9F27',
  step3Dim: 'rgba(239,159,39,0.15)',

  // Ripeness colors
  fullyRipened: '#FF5078',
  halfRipened: '#5078FF',
  green: '#50C850',
}

export const light = {
  // Backgrounds
  bg: '#f5f5f5',
  bgSecondary: '#ffffff',
  card: '#ffffff',
  cardAlt: '#f9f9f9',
  cardElevated: '#ffffff',

  // Borders
  border: '#e8e8e8',
  borderLight: '#efefef',
  borderMid: '#e0e0e0',

  // Text
  textPrimary: '#111111',
  textSec: '#666',
  textTertiary: '#999',
  textMuted: '#c0c0c0',
  infoText: '#888',

  // Status bar
  statusBar: 'dark-content',

  // UI elements
  switchTrackOff: '#d0d0d0',
  hudBg: 'rgba(255,255,255,0.88)',
  analyzingBg: 'rgba(255,255,255,0.7)',
  analyzingCard: '#fff',
  scanLine: 'rgba(0,0,0,0.06)',

  // Buttons
  btnText: '#ffffff',
  btnSecBg: '#f0f0f0',

  // Accent
  accent: '#1D9E75',
  accentDim: 'rgba(29,158,117,0.1)',
  accentBorder: 'rgba(29,158,117,0.3)',

  // Step colors
  step1: '#378ADD',
  step1Dim: 'rgba(55,138,221,0.1)',
  step2: '#1D9E75',
  step2Dim: 'rgba(29,158,117,0.1)',
  step3: '#EF9F27',
  step3Dim: 'rgba(239,159,39,0.1)',

  // Ripeness colors
  fullyRipened: '#FF5078',
  halfRipened: '#5078FF',
  green: '#50C850',
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