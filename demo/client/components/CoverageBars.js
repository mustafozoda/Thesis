import React, { useEffect, useRef } from 'react'
import { View, Text, StyleSheet, Animated } from 'react-native'
import { useTheme } from '../context/ThemeContext'

const RIPENESS_META = {
  fully_ripened: { color: '#FF5078', label: 'Fully Ripened', emoji: '🔴' },
  half_ripened: { color: '#5078FF', label: 'Half Ripened', emoji: '🟡' },
  green: { color: '#50C850', label: 'Green', emoji: '🟢' },
}

function AnimatedBar({ pct, color, delay = 0 }) {
  const widthAnim = useRef(new Animated.Value(0)).current
  useEffect(() => {
    Animated.timing(widthAnim, {
      toValue: Math.min(pct, 100),
      duration: 600,
      delay,
      useNativeDriver: false,
    }).start()
  }, [pct, delay, widthAnim])

  return (
    <Animated.View
      style={{
        height: '100%',
        borderRadius: 3,
        backgroundColor: color,
        width: widthAnim.interpolate({
          inputRange: [0, 100],
          outputRange: ['0%', '100%'],
        }),
      }}
    />
  )
}

export default function CoverageBars({ coverage, confidence }) {
  const { theme: t } = useTheme()
  const s = makeStyles(t)

  if (!coverage) return null

  const b_full = coverage.b_fully_ripened ?? 0
  const b_half = coverage.b_half_ripened ?? 0
  const b_green = coverage.b_green ?? 0
  const l_full = coverage.l_fully_ripened ?? 0
  const l_half = coverage.l_half_ripened ?? 0
  const l_green = coverage.l_green ?? 0
  const bg = coverage.background ?? 0

  const tomatoTotal = b_full + b_half + b_green + l_full + l_half + l_green
  if (tomatoTotal < 0.5) return null

  const norm = (v) => (v / tomatoTotal) * 100

  const ripeness = [
    ['fully_ripened', norm(b_full + l_full)],
    ['half_ripened', norm(b_half + l_half)],
    ['green', norm(b_green + l_green)],
  ]
    .filter(([, v]) => v > 0.3)
    .sort((a, b) => b[1] - a[1])

  const dominant = ripeness[0]
  const dominantMeta = dominant ? RIPENESS_META[dominant[0]] : null

  // Confidence for each ripeness group
  const getConfidence = (key) => {
    if (!confidence) return null
    if (key === 'fully_ripened') {
      const vals = [confidence.b_fully_ripened, confidence.l_fully_ripened].filter(v => v > 0)
      return vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null
    }
    if (key === 'half_ripened') {
      const vals = [confidence.b_half_ripened, confidence.l_half_ripened].filter(v => v > 0)
      return vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null
    }
    if (key === 'green') {
      const vals = [confidence.b_green, confidence.l_green].filter(v => v > 0)
      return vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null
    }
    return null
  }

  return (
    <View style={s.container}>
      {/* Dominant result badge */}
      {dominantMeta && (
        <View style={[s.dominantBadge, { borderColor: dominantMeta.color + '40', backgroundColor: dominantMeta.color + '12' }]}>
          <View style={[s.dominantDot, { backgroundColor: dominantMeta.color }]} />
          <Text style={[s.dominantLabel, { color: dominantMeta.color }]}>
            {dominantMeta.label}
          </Text>
          <Text style={s.dominantPct}>{dominant[1].toFixed(0)}%</Text>
        </View>
      )}

      {/* Stacked bar overview */}
      <View style={s.stackedBarContainer}>
        <View style={s.stackedBar}>
          {ripeness.map(([cls, pct]) => (
            <View
              key={cls}
              style={[
                s.stackSegment,
                {
                  flex: pct,
                  backgroundColor: RIPENESS_META[cls]?.color ?? '#888',
                },
              ]}
            />
          ))}
        </View>
        <Text style={s.totalLabel}>{tomatoTotal.toFixed(1)}% of frame</Text>
      </View>

      {/* Individual bars */}
      <View style={s.barsSection}>
        <Text style={s.sectionTitle}>RIPENESS BREAKDOWN</Text>
        {ripeness.map(([cls, pct], i) => {
          const m = RIPENESS_META[cls] ?? { color: '#888', label: cls }
          const conf = getConfidence(cls)
          return (
            <View key={cls} style={s.barRow}>
              <View style={[s.colorDot, { backgroundColor: m.color }]} />
              <View style={s.barInfo}>
                <View style={s.barLabelRow}>
                  <Text style={s.barLabel}>{m.label}</Text>
                  <View style={s.barRightInfo}>
                    {conf !== null && (
                      <Text style={[s.confText, { color: m.color }]}>{conf}% conf</Text>
                    )}
                    <Text style={s.barPct}>{pct.toFixed(1)}%</Text>
                  </View>
                </View>
                <View style={s.trackBg}>
                  <AnimatedBar pct={pct} color={m.color} delay={i * 80} />
                </View>
              </View>
            </View>
          )
        })}
      </View>

      {/* Background coverage footnote */}
      {bg > 0 && (
        <Text style={s.bgNote}>Background: {bg.toFixed(1)}%</Text>
      )}
    </View>
  )
}

const makeStyles = (t) => StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  dominantBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    marginBottom: 12,
    gap: 7,
  },
  dominantDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  dominantLabel: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  dominantPct: {
    fontSize: 12,
    fontWeight: '700',
    color: t.textPrimary,
  },
  stackedBarContainer: {
    marginBottom: 14,
  },
  stackedBar: {
    flexDirection: 'row',
    height: 6,
    borderRadius: 3,
    overflow: 'hidden',
    backgroundColor: t.borderLight,
    marginBottom: 5,
  },
  stackSegment: {
    height: '100%',
  },
  totalLabel: {
    fontSize: 10,
    color: t.textTertiary,
    letterSpacing: 0.3,
  },
  barsSection: {
    gap: 10,
  },
  sectionTitle: {
    fontSize: 9,
    color: t.textMuted,
    fontWeight: '700',
    letterSpacing: 1.4,
    marginBottom: 4,
  },
  barRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  colorDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 2,
  },
  barInfo: {
    flex: 1,
    gap: 4,
  },
  barLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  barLabel: {
    fontSize: 11,
    color: t.textSec,
    fontWeight: '500',
  },
  barRightInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  confText: {
    fontSize: 10,
    fontWeight: '600',
    opacity: 0.8,
  },
  barPct: {
    fontSize: 11,
    color: t.textPrimary,
    fontWeight: '700',
    minWidth: 38,
    textAlign: 'right',
  },
  trackBg: {
    height: 4,
    backgroundColor: t.borderLight,
    borderRadius: 3,
    overflow: 'hidden',
  },
  bgNote: {
    fontSize: 10,
    color: t.textTertiary,
    marginTop: 10,
    textAlign: 'right',
  },
})