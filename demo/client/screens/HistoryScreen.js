import React, { useState } from 'react'
import {
  View, Text, StyleSheet, FlatList, TouchableOpacity,
  StatusBar, Alert, Dimensions, Image,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTheme } from '../context/ThemeContext'
import { useScanHistory } from '../context/ScanHistoryContext'

const { width: W } = Dimensions.get('window')

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

function formatTime(ts) {
  const d = new Date(ts)
  const now = new Date()
  const diffMs = now - d
  const diffMin = Math.floor(diffMs / 60000)
  const diffHr = Math.floor(diffMs / 3600000)
  const diffDay = Math.floor(diffMs / 86400000)

  if (diffMin < 1) return 'Just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHr < 24) return `${diffHr}h ago`
  if (diffDay < 7) return `${diffDay}d ago`
  return d.toLocaleDateString()
}

function formatTimeFull(ts) {
  const d = new Date(ts)
  return d.toLocaleString()
}

function getDominant(coverage) {
  if (!coverage) return null
  const b_full = (coverage.b_fully_ripened ?? 0)
  const b_half = (coverage.b_half_ripened ?? 0)
  const b_green = (coverage.b_green ?? 0)
  const l_full = (coverage.l_fully_ripened ?? 0)
  const l_half = (coverage.l_half_ripened ?? 0)
  const l_green = (coverage.l_green ?? 0)
  const groups = {
    fully_ripened: b_full + l_full,
    half_ripened: b_half + l_half,
    green: b_green + l_green,
  }
  const dominant = Object.entries(groups).sort((a, b) => b[1] - a[1])[0]
  return dominant[1] > 0.3 ? dominant : null
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

function MiniBar({ coverage }) {
  const total = getTomatoTotal(coverage)
  if (total < 0.5) return null
  const norm = v => (v / total) * 100
  const segments = [
    { key: 'fully_ripened', val: norm((coverage.b_fully_ripened ?? 0) + (coverage.l_fully_ripened ?? 0)) },
    { key: 'half_ripened', val: norm((coverage.b_half_ripened ?? 0) + (coverage.l_half_ripened ?? 0)) },
    { key: 'green', val: norm((coverage.b_green ?? 0) + (coverage.l_green ?? 0)) },
  ].filter(s => s.val > 0.5)

  return (
    <View style={{ flexDirection: 'row', height: 4, borderRadius: 2, overflow: 'hidden', flex: 1, backgroundColor: '#1e1e1e' }}>
      {segments.map(s => (
        <View key={s.key} style={{ flex: s.val, backgroundColor: RIPENESS_COLORS[s.key] }} />
      ))}
    </View>
  )
}

function ScanCard({ entry, onPress, onDelete, t }) {
  const s = cardStyles(t)
  const dominant = getDominant(entry.coverage)
  const total = getTomatoTotal(entry.coverage).toFixed(1)
  const stepKey = Object.keys(STEP_COLORS).find(k => entry.model?.includes(k))
  const stepColor = STEP_COLORS[stepKey] ?? '#888'
  const encoder = entry.model?.includes('EfficientNet') ? 'EffNet-B0' : 'MobileV2'
  const mode = entry.mode ?? 'photo'
  const modeIcon = mode === 'live' ? '◉' : mode === 'upload' ? '↑' : '⬡'

  return (
    <TouchableOpacity
      style={[s.card, { backgroundColor: t.card, borderColor: t.border }]}
      onPress={onPress}
      onLongPress={onDelete}
      activeOpacity={0.75}
    >
      {/* Thumbnail */}
      <View style={s.thumbWrap}>
        {entry.overlayB64 ? (
          <Image
            source={{ uri: `data:image/jpeg;base64,${entry.overlayB64}` }}
            style={s.thumb}
            resizeMode="cover"
          />
        ) : (
          <View style={[s.thumb, s.thumbPlaceholder, { backgroundColor: t.cardAlt }]}>
            <Text style={{ fontSize: 22 }}>🍅</Text>
          </View>
        )}
        {/* Mode badge */}
        <View style={[s.modeBadge, { backgroundColor: t.bg + 'cc' }]}>
          <Text style={[s.modeBadgeText, { color: t.textSec }]}>{modeIcon}</Text>
        </View>
      </View>

      {/* Info */}
      <View style={s.info}>
        {/* Top row */}
        <View style={s.topRow}>
          <View style={[s.stepPill, { backgroundColor: stepColor + '22' }]}>
            <Text style={[s.stepPillText, { color: stepColor }]}>{stepKey ?? 'S?'}</Text>
          </View>
          <Text style={[s.encoder, { color: t.textTertiary }]}>{encoder}</Text>
          <Text style={[s.timeText, { color: t.textMuted }]}>{formatTime(entry.timestamp)}</Text>
        </View>

        {/* Dominant */}
        {dominant ? (
          <View style={s.dominantRow}>
            <View style={[s.dominantDot, { backgroundColor: RIPENESS_COLORS[dominant[0]] }]} />
            <Text style={[s.dominantText, { color: t.textPrimary }]}>
              {dominant[0].replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}
            </Text>
            <Text style={[s.totalText, { color: t.textSec }]}>{total}% tomatoes</Text>
          </View>
        ) : (
          <Text style={[s.noTomato, { color: t.textTertiary }]}>No tomatoes detected</Text>
        )}

        {/* Mini bar */}
        <View style={s.barRow}>
          <MiniBar coverage={entry.coverage} />
          {entry.latency && (
            <Text style={[s.latency, { color: t.textMuted }]}>{entry.latency}ms</Text>
          )}
        </View>
      </View>
    </TouchableOpacity>
  )
}

const cardStyles = (t) => StyleSheet.create({
  card: {
    flexDirection: 'row',
    borderRadius: 14,
    borderWidth: 1,
    marginBottom: 10,
    overflow: 'hidden',
  },
  thumbWrap: { position: 'relative' },
  thumb: { width: 90, height: 90 },
  thumbPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  modeBadge: {
    position: 'absolute',
    bottom: 4,
    left: 4,
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 5,
  },
  modeBadgeText: { fontSize: 10 },
  info: {
    flex: 1,
    padding: 11,
    justifyContent: 'space-between',
    gap: 5,
  },
  topRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  stepPill: { paddingHorizontal: 7, paddingVertical: 3, borderRadius: 6 },
  stepPillText: { fontSize: 10, fontWeight: '800' },
  encoder: { fontSize: 11, flex: 1 },
  timeText: { fontSize: 10 },
  dominantRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dominantDot: { width: 7, height: 7, borderRadius: 3.5 },
  dominantText: { fontSize: 13, fontWeight: '600', flex: 1 },
  totalText: { fontSize: 11 },
  noTomato: { fontSize: 12 },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  latency: { fontSize: 10, flexShrink: 0 },
})

// ── Detail Modal ──────────────────────────────────────────────────────────────
function DetailModal({ entry, onClose, t }) {
  const s = modalStyles(t)
  if (!entry) return null
  const dominant = getDominant(entry.coverage)
  const total = getTomatoTotal(entry.coverage)
  const stepKey = Object.keys(STEP_COLORS).find(k => entry.model?.includes(k))
  const stepColor = STEP_COLORS[stepKey] ?? '#888'

  const ripenessGroups = [
    { key: 'fully_ripened', label: 'Fully Ripened', val: ((entry.coverage?.b_fully_ripened ?? 0) + (entry.coverage?.l_fully_ripened ?? 0)) },
    { key: 'half_ripened', label: 'Half Ripened', val: ((entry.coverage?.b_half_ripened ?? 0) + (entry.coverage?.l_half_ripened ?? 0)) },
    { key: 'green', label: 'Green', val: ((entry.coverage?.b_green ?? 0) + (entry.coverage?.l_green ?? 0)) },
  ].filter(g => g.val > 0.1)

  return (
    <View style={s.overlay}>
      <TouchableOpacity style={s.backdrop} onPress={onClose} activeOpacity={1} />
      <View style={[s.modal, { backgroundColor: t.card, borderColor: t.border }]}>
        {/* Image */}
        {entry.overlayB64 && (
          <Image
            source={{ uri: `data:image/jpeg;base64,${entry.overlayB64}` }}
            style={s.image}
            resizeMode="cover"
          />
        )}

        <View style={s.body}>
          {/* Header */}
          <View style={s.modalHeader}>
            <View>
              <Text style={[s.modalTitle, { color: t.textPrimary }]}>Scan Detail</Text>
              <Text style={[s.modalTime, { color: t.textSec }]}>{formatTimeFull(entry.timestamp)}</Text>
            </View>
            <TouchableOpacity onPress={onClose} style={[s.closeBtn, { backgroundColor: t.cardAlt }]}>
              <Text style={[s.closeText, { color: t.textPrimary }]}>✕</Text>
            </TouchableOpacity>
          </View>

          {/* Model info */}
          <View style={[s.infoRow, { backgroundColor: t.cardAlt, borderColor: t.border }]}>
            <View style={[s.stepPill, { backgroundColor: stepColor + '22' }]}>
              <Text style={[s.stepPillText, { color: stepColor }]}>{stepKey}</Text>
            </View>
            <Text style={[s.infoText, { color: t.textSec }]}>
              {entry.model?.includes('EfficientNet') ? 'EfficientNet-B0' : 'MobileNetV2'}
            </Text>
            {entry.latency && (
              <Text style={[s.infoText, { color: t.textMuted }]}>⚡ {entry.latency}ms</Text>
            )}
          </View>

          {/* Ripeness */}
          {ripenessGroups.length > 0 && (
            <View style={s.section}>
              <Text style={[s.sectionLabel, { color: t.textMuted }]}>RIPENESS</Text>
              {ripenessGroups.map(g => (
                <View key={g.key} style={s.rRow}>
                  <View style={[s.rDot, { backgroundColor: RIPENESS_COLORS[g.key] }]} />
                  <Text style={[s.rLabel, { color: t.textSec }]}>{g.label}</Text>
                  <View style={[s.rTrack, { backgroundColor: t.borderLight }]}>
                    <View style={[s.rFill, {
                      width: `${total > 0 ? (g.val / total) * 100 : 0}%`,
                      backgroundColor: RIPENESS_COLORS[g.key],
                    }]} />
                  </View>
                  <Text style={[s.rPct, { color: t.textPrimary }]}>{g.val.toFixed(1)}%</Text>
                </View>
              ))}
            </View>
          )}

          {/* Total */}
          <View style={[s.totalRow, { borderTopColor: t.border }]}>
            <Text style={[s.totalLabel, { color: t.textSec }]}>Total tomato coverage</Text>
            <Text style={[s.totalVal, { color: t.textPrimary }]}>{total.toFixed(1)}%</Text>
          </View>
        </View>
      </View>
    </View>
  )
}

const modalStyles = (t) => StyleSheet.create({
  overlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'flex-end', zIndex: 100 },
  backdrop: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.6)' },
  modal: {
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    borderWidth: 1,
    borderBottomWidth: 0,
    overflow: 'hidden',
    maxHeight: '85%',
  },
  image: { width: '100%', height: 220 },
  body: { padding: 18, gap: 14 },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  modalTitle: { fontSize: 18, fontWeight: '700', letterSpacing: -0.3 },
  modalTime: { fontSize: 12, marginTop: 2 },
  closeBtn: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  closeText: { fontSize: 14, fontWeight: '600' },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
  },
  stepPill: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6 },
  stepPillText: { fontSize: 11, fontWeight: '800' },
  infoText: { fontSize: 12, flex: 1 },
  section: { gap: 8 },
  sectionLabel: { fontSize: 9, fontWeight: '700', letterSpacing: 1.3 },
  rRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  rDot: { width: 8, height: 8, borderRadius: 4 },
  rLabel: { width: 100, fontSize: 12 },
  rTrack: { flex: 1, height: 4, borderRadius: 2, overflow: 'hidden' },
  rFill: { height: '100%', borderRadius: 2 },
  rPct: { width: 40, fontSize: 12, fontWeight: '700', textAlign: 'right' },
  totalRow: { flexDirection: 'row', justifyContent: 'space-between', paddingTop: 12, borderTopWidth: 1 },
  totalLabel: { fontSize: 13 },
  totalVal: { fontSize: 13, fontWeight: '700' },
})

// ── Main Screen ───────────────────────────────────────────────────────────────
export default function HistoryScreen({ navigation }) {
  const { theme: t } = useTheme()
  const { history, clearHistory, removeEntry } = useScanHistory()
  const s = makeStyles(t)
  const [selected, setSelected] = useState(null)

  const handleDelete = (id) => {
    Alert.alert('Delete Scan', 'Remove this scan from history?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => removeEntry(id) },
    ])
  }

  const handleClearAll = () => {
    Alert.alert('Clear History', 'Remove all scan history?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Clear All', style: 'destructive', onPress: clearHistory },
    ])
  }

  return (
    <SafeAreaView style={[s.root, { backgroundColor: t.bg }]}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={s.backBtn}>
          <Text style={[s.backArrow, { color: t.textPrimary }]}>←</Text>
        </TouchableOpacity>
        <Text style={[s.headerTitle, { color: t.textPrimary }]}>Scan History</Text>
        {history.length > 0 ? (
          <TouchableOpacity onPress={handleClearAll} style={s.clearBtn}>
            <Text style={[s.clearText, { color: '#FF5078' }]}>Clear</Text>
          </TouchableOpacity>
        ) : (
          <View style={{ width: 50 }} />
        )}
      </View>

      {/* Count badge */}
      {history.length > 0 && (
        <View style={s.countRow}>
          <View style={[s.countBadge, { backgroundColor: t.accentDim, borderColor: t.accentBorder }]}>
            <Text style={[s.countText, { color: t.accent }]}>{history.length} scans</Text>
          </View>
          <Text style={[s.hintText, { color: t.textMuted }]}>Long press to delete</Text>
        </View>
      )}

      {/* Empty state */}
      {history.length === 0 && (
        <View style={s.empty}>
          <View style={[s.emptyIcon, { backgroundColor: t.card, borderColor: t.border }]}>
            <Text style={s.emptyEmoji}>🍅</Text>
          </View>
          <Text style={[s.emptyTitle, { color: t.textPrimary }]}>No scans yet</Text>
          <Text style={[s.emptySub, { color: t.textSec }]}>
            Your scan results will appear here after you analyze tomato images.
          </Text>
        </View>
      )}

      {/* List */}
      <FlatList
        data={history}
        keyExtractor={item => item.id}
        contentContainerStyle={s.list}
        showsVerticalScrollIndicator={false}
        renderItem={({ item }) => (
          <ScanCard
            entry={item}
            t={t}
            onPress={() => setSelected(item)}
            onDelete={() => handleDelete(item.id)}
          />
        )}
      />

      {/* Detail modal */}
      {selected && (
        <DetailModal entry={selected} onClose={() => setSelected(null)} t={t} />
      )}
    </SafeAreaView>
  )
}

const makeStyles = (t) => StyleSheet.create({
  root: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  backBtn: { padding: 4, width: 50 },
  backArrow: { fontSize: 22 },
  headerTitle: { fontSize: 17, fontWeight: '700', letterSpacing: -0.3 },
  clearBtn: { width: 50, alignItems: 'flex-end' },
  clearText: { fontSize: 14, fontWeight: '600' },
  countRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    marginBottom: 8,
    gap: 10,
  },
  countBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
    borderWidth: 1,
  },
  countText: { fontSize: 12, fontWeight: '600' },
  hintText: { fontSize: 11 },
  list: { paddingHorizontal: 16, paddingBottom: 24 },
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    gap: 14,
  },
  emptyIcon: {
    width: 80,
    height: 80,
    borderRadius: 24,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  emptyEmoji: { fontSize: 36 },
  emptyTitle: { fontSize: 20, fontWeight: '700', letterSpacing: -0.3 },
  emptySub: { fontSize: 14, textAlign: 'center', lineHeight: 22 },
})