import React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import { useTheme } from '../context/ThemeContext'

const RIPENESS_META = {
  fully_ripened: { color: '#FF5078', label: 'Fully ripened' },
  half_ripened: { color: '#5078FF', label: 'Half ripened' },
  green: { color: '#50C850', label: 'Green' },
}

export default function CoverageBars({ coverage }) {
  const { theme: t } = useTheme()
  const s = makeStyles(t)

  if (!coverage) return null

  const b_full = coverage.b_fully_ripened ?? 0
  const b_half = coverage.b_half_ripened ?? 0
  const b_green = coverage.b_green ?? 0
  const l_full = coverage.l_fully_ripened ?? 0
  const l_half = coverage.l_half_ripened ?? 0
  const l_green = coverage.l_green ?? 0

  const tomatoTotal = b_full + b_half + b_green + l_full + l_half + l_green
  if (tomatoTotal < 0.5) return null

  const norm = (v) => (v / tomatoTotal) * 100

  const ripeness = [
    ['fully_ripened', norm(b_full + l_full)],
    ['half_ripened', norm(b_half + l_half)],
    ['green', norm(b_green + l_green)],
  ]
    .filter(([, v]) => v > 0.5)
    .sort((a, b) => b[1] - a[1])

  const Section = ({ title, data, meta }) => (
    <View style={s.section}>
      <Text style={s.sectionTitle}>{title}</Text>
      <View style={s.stackedBar}>
        {data.map(([cls, pct]) => (
          <View
            key={cls}
            style={[s.stackSegment, {
              flex: pct,
              backgroundColor: meta[cls]?.color ?? '#888',
            }]}
          />
        ))}
      </View>
      {data.map(([cls, pct]) => {
        const m = meta[cls] ?? { color: '#888', label: cls }
        return (
          <View key={cls} style={s.row}>
            <View style={[s.dot, { backgroundColor: m.color }]} />
            <Text style={s.label} numberOfLines={1}>{m.label}</Text>
            <View style={s.track}>
              <View style={[s.fill, {
                width: `${Math.min(pct, 100)}%`,
                backgroundColor: m.color,
              }]} />
            </View>
            <Text style={s.pct}>{pct.toFixed(1)}%</Text>
          </View>
        )
      })}
    </View>
  )

  return (
    <View style={s.container}>
      {ripeness.length > 0 && (
        <Section title="RIPENESS" data={ripeness} meta={RIPENESS_META} />
      )}
    </View>
  )
}

const makeStyles = (t) => StyleSheet.create({
  container: { paddingHorizontal: 16, paddingVertical: 6 },
  section: { marginBottom: 10 },
  sectionTitle: {
    fontSize: 10, color: t.textMuted, fontWeight: '700',
    letterSpacing: 1.2, marginBottom: 6,
  },
  stackedBar: {
    flexDirection: 'row', height: 4, borderRadius: 2,
    overflow: 'hidden', marginBottom: 8, backgroundColor: t.borderLight,
  },
  stackSegment: { height: '100%' },
  row: { flexDirection: 'row', alignItems: 'center', marginBottom: 4, gap: 8 },
  dot: { width: 7, height: 7, borderRadius: 3.5 },
  label: { width: 110, fontSize: 11, color: t.textSec },
  track: { flex: 1, height: 4, backgroundColor: t.borderLight, borderRadius: 2, overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 2 },
  pct: { width: 42, fontSize: 11, color: t.textPrimary, textAlign: 'right', fontWeight: '600' },
})