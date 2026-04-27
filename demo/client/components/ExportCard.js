import React, { useRef, useState } from 'react'
import {
  View, Text, StyleSheet, TouchableOpacity,
  Image, Alert, ActivityIndicator, Dimensions,
} from 'react-native'
import * as Sharing from 'expo-sharing'
import * as MediaLibrary from 'expo-media-library'
import ViewShot from 'react-native-view-shot'
import { useTheme } from '../context/ThemeContext'

const { width: W } = Dimensions.get('window')
const CARD_W = W - 32

const RIPENESS_COLORS = {
  fully_ripened: '#FF5078',
  half_ripened: '#5078FF',
  green: '#50C850',
}

const STEP_COLORS = {
  Step1: '#378ADD',
  Step2: '#1D9E75',
  Step3: '#EF9F27',
}

function getTomatoTotal(coverage) {
  if (!coverage) return 0
  return (
    (coverage.b_fully_ripened ?? 0) +
    (coverage.b_half_ripened ?? 0) +
    (coverage.b_green ?? 0) +
    (coverage.l_fully_ripened ?? 0) +
    (coverage.l_half_ripened ?? 0) +
    (coverage.l_green ?? 0)
  )
}

// The card that gets captured as an image
function SummaryCard({ overlayB64, coverage, confidence, model, latency, inferenceMs, cardRef }) {
  const total = getTomatoTotal(coverage)
  const stepKey = Object.keys(STEP_COLORS).find(k => model?.includes(k))
  const stepColor = STEP_COLORS[stepKey] ?? '#888'
  const encoder = model?.includes('EfficientNet') ? 'EfficientNet-B0' : 'MobileNetV2'

  const ripenessGroups = [
    {
      key: 'fully_ripened',
      label: 'Fully Ripened',
      val: (coverage?.b_fully_ripened ?? 0) + (coverage?.l_fully_ripened ?? 0),
    },
    {
      key: 'half_ripened',
      label: 'Half Ripened',
      val: (coverage?.b_half_ripened ?? 0) + (coverage?.l_half_ripened ?? 0),
    },
    {
      key: 'green',
      label: 'Green',
      val: (coverage?.b_green ?? 0) + (coverage?.l_green ?? 0),
    },
  ].filter(g => g.val > 0.1).sort((a, b) => b.val - a.val)

  const dominant = ripenessGroups[0]

  return (
    <ViewShot ref={cardRef} options={{ format: 'jpg', quality: 0.95 }}>
      <View style={cs.card}>
        {/* Overlay image */}
        {overlayB64 && (
          <Image
            source={{ uri: `data:image/jpeg;base64,${overlayB64}` }}
            style={cs.image}
            resizeMode="cover"
          />
        )}

        {/* Content */}
        <View style={cs.content}>
          {/* Header */}
          <View style={cs.header}>
            <View style={cs.logoRow}>
              <View style={cs.logoDot} />
              <Text style={cs.logoText}>Tomato</Text>
            </View>
            <Text style={cs.dateText}>{new Date().toLocaleDateString()}</Text>
          </View>

          {/* Dominant result */}
          {dominant && (
            <View style={[cs.dominantBadge, { borderColor: RIPENESS_COLORS[dominant.key] + '50', backgroundColor: RIPENESS_COLORS[dominant.key] + '15' }]}>
              <View style={[cs.dominantDot, { backgroundColor: RIPENESS_COLORS[dominant.key] }]} />
              <Text style={[cs.dominantLabel, { color: RIPENESS_COLORS[dominant.key] }]}>
                {dominant.label}
              </Text>
              <Text style={cs.dominantPct}>{total > 0 ? ((dominant.val / total) * 100).toFixed(0) : 0}%</Text>
            </View>
          )}

          {/* Ripeness bars */}
          {ripenessGroups.length > 0 && (
            <View style={cs.barsSection}>
              {ripenessGroups.map(g => (
                <View key={g.key} style={cs.barRow}>
                  <View style={[cs.barDot, { backgroundColor: RIPENESS_COLORS[g.key] }]} />
                  <Text style={cs.barLabel}>{g.label}</Text>
                  <View style={cs.barTrack}>
                    <View style={[cs.barFill, {
                      width: `${total > 0 ? (g.val / total) * 100 : 0}%`,
                      backgroundColor: RIPENESS_COLORS[g.key],
                    }]} />
                  </View>
                  <Text style={cs.barPct}>{g.val.toFixed(1)}%</Text>
                </View>
              ))}
            </View>
          )}

          {/* Stats row */}
          <View style={cs.statsRow}>
            <View style={cs.statItem}>
              <Text style={cs.statVal}>{total.toFixed(1)}%</Text>
              <Text style={cs.statLabel}>Coverage</Text>
            </View>
            <View style={cs.statDivider} />
            <View style={cs.statItem}>
              <View style={[cs.modelPill, { backgroundColor: stepColor + '22' }]}>
                <Text style={[cs.modelPillText, { color: stepColor }]}>{stepKey}</Text>
              </View>
              <Text style={cs.statLabel}>{encoder}</Text>
            </View>
            {(latency || inferenceMs) && (
              <>
                <View style={cs.statDivider} />
                <View style={cs.statItem}>
                  <Text style={cs.statVal}>{inferenceMs ?? latency}ms</Text>
                  <Text style={cs.statLabel}>Inference</Text>
                </View>
              </>
            )}
          </View>

          {/* Footer */}
          <Text style={cs.footer}>Tomato Quality Segmentation v2.0</Text>
        </View>
      </View>
    </ViewShot>
  )
}

// Dark card styles
const cs = StyleSheet.create({
  card: {
    width: CARD_W,
    backgroundColor: '#0e0e0e',
    borderRadius: 18,
    overflow: 'hidden',
  },
  image: {
    width: CARD_W,
    height: CARD_W * 0.65,
  },
  content: {
    padding: 16,
    gap: 12,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  logoDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#1D9E75' },
  logoText: { fontSize: 14, fontWeight: '700', color: '#f0f0f0', letterSpacing: -0.3 },
  dateText: { fontSize: 11, color: '#555' },
  dominantBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    gap: 7,
  },
  dominantDot: { width: 8, height: 8, borderRadius: 4 },
  dominantLabel: { fontSize: 13, fontWeight: '700' },
  dominantPct: { fontSize: 13, fontWeight: '700', color: '#f0f0f0' },
  barsSection: { gap: 7 },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  barDot: { width: 7, height: 7, borderRadius: 3.5 },
  barLabel: { width: 100, fontSize: 11, color: '#888' },
  barTrack: { flex: 1, height: 3, backgroundColor: '#222', borderRadius: 2, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 2 },
  barPct: { width: 38, fontSize: 11, fontWeight: '700', color: '#f0f0f0', textAlign: 'right' },
  statsRow: {
    flexDirection: 'row',
    backgroundColor: '#141414',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
  },
  statItem: { flex: 1, alignItems: 'center', gap: 3 },
  statVal: { fontSize: 15, fontWeight: '700', color: '#f0f0f0' },
  statLabel: { fontSize: 10, color: '#555', letterSpacing: 0.3 },
  statDivider: { width: 1, height: 28, backgroundColor: '#222' },
  modelPill: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  modelPillText: { fontSize: 11, fontWeight: '800' },
  footer: { fontSize: 10, color: '#333', textAlign: 'center', letterSpacing: 0.5 },
})

// ── Export Button ─────────────────────────────────────────────────────────────
export default function ExportCard({ overlayB64, coverage, confidence, model, latency, inferenceMs }) {
  const { theme: t } = useTheme()
  const cardRef = useRef(null)
  const [saving, setSaving] = useState(false)

  const handleExport = async () => {
    setSaving(true)
    try {
      const uri = await cardRef.current.capture()

      // Try to save to gallery
      const { status } = await MediaLibrary.requestPermissionsAsync()
      if (status === 'granted') {
        await MediaLibrary.saveToLibraryAsync(uri)
        Alert.alert(
          'Saved!',
          'Scan summary saved to your photo library.',
          [
            { text: 'Also Share', onPress: () => Sharing.shareAsync(uri, { mimeType: 'image/jpeg' }) },
            { text: 'OK', style: 'cancel' },
          ]
        )
      } else {
        await Sharing.shareAsync(uri, {
          mimeType: 'image/jpeg',
          dialogTitle: 'Save or share your scan result',
        })
      }
    } catch (e) {
      Alert.alert('Export failed', e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <View style={es.wrapper}>
      {/* Hidden card for capture — rendered off-screen */}
      <View style={es.offscreen}>
        <SummaryCard
          cardRef={cardRef}
          overlayB64={overlayB64}
          coverage={coverage}
          confidence={confidence}
          model={model}
          latency={latency}
          inferenceMs={inferenceMs}
        />
      </View>

      {/* Export button */}
      <TouchableOpacity
        style={[es.exportBtn, { backgroundColor: t.cardAlt, borderColor: t.border }]}
        onPress={handleExport}
        disabled={saving}
        activeOpacity={0.8}
      >
        {saving
          ? <ActivityIndicator size="small" color={t.accent} />
          : <Text style={[es.exportIcon, { color: t.accent }]}>↓</Text>
        }
        <Text style={[es.exportText, { color: t.textPrimary }]}>
          {saving ? 'Saving...' : 'Save & Share'}
        </Text>
      </TouchableOpacity>
    </View>
  )
}

const es = StyleSheet.create({
  wrapper: { position: 'relative' },
  offscreen: {
    position: 'absolute',
    top: -9999,
    left: -9999,
    opacity: 0,
  },
  exportBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingVertical: 13,
    paddingHorizontal: 18,
    borderRadius: 13,
    borderWidth: 1,
  },
  exportIcon: { fontSize: 16, fontWeight: '700' },
  exportText: { fontSize: 14, fontWeight: '600' },
})