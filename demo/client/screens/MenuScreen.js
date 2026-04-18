import React, { useState, useEffect, useRef } from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import {
  View, Text, TouchableOpacity, FlatList,
  StyleSheet, ActivityIndicator, StatusBar,
  Animated, Image,
} from 'react-native'
import { useTheme } from '../context/ThemeContext'
import { useServer } from '../context/ServerContext'
import { useScanHistory } from '../context/ScanHistoryContext'

const STEP_META = {
  Step1: { color: '#378ADD', label: 'Baseline', desc: 'Natural background', badge: 'S1' },
  Step2: { color: '#1D9E75', label: 'Best', desc: 'Background removed', badge: 'S2' },
  Step3: { color: '#EF9F27', label: 'Synthetic', desc: 'Synthetic background', badge: 'S3' },
}

const STEP_INFO = {
  Step1: 'Baseline model trained on natural farm backgrounds. mIoU 0.64.',
  Step2: 'Best performer — background removed during training. mIoU 0.76.',
  Step3: 'Robust model trained with procedural synthetic backgrounds. mIoU 0.72.',
}

const MODES = [
  { key: 'live', icon: '◉', label: 'Live', sub: 'Real-time', color: '#1D9E75' },
  { key: 'photo', icon: '⬡', label: 'Photo', sub: 'Capture', color: '#378ADD' },
  { key: 'upload', icon: '↑', label: 'Upload', sub: 'Gallery', color: '#EF9F27' },
]

export default function MenuScreen({ navigation }) {
  const { theme: t, isDark, toggle } = useTheme()
  const { server } = useServer()
  const { history } = useScanHistory()
  const s = makeStyles(t)

  const [models, setModels] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const [retrying, setRetrying] = useState(false)

  const fadeAnim = useRef(new Animated.Value(0)).current
  const scaleAnim = useRef(new Animated.Value(1)).current

  useEffect(() => {
    fetchModels()
  }, [server]) // eslint-disable-line react-hooks/exhaustive-deps

  const fetchModels = () => {
    setLoading(true)
    setError(false)
    fetch(`${server}/models`)
      .then(r => r.json())
      .then(d => {
        setModels(d.models)
        setSelected(d.models[1] ?? d.models[0])
        setLoading(false)
        Animated.timing(fadeAnim, { toValue: 1, duration: 400, useNativeDriver: true }).start()
      })
      .catch(() => {
        setError(true)
        setLoading(false)
      })
  }

  const handleRetry = () => {
    setRetrying(true)
    setTimeout(() => { setRetrying(false); fetchModels() }, 500)
  }

  const handleThemeToggle = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, { toValue: 0.7, duration: 100, useNativeDriver: true }),
      Animated.timing(scaleAnim, { toValue: 1.2, duration: 140, useNativeDriver: true }),
      Animated.timing(scaleAnim, { toValue: 1.0, duration: 100, useNativeDriver: true }),
    ]).start()
    toggle()
  }

  const go = (mode) => {
    if (!selected) return
    navigation.navigate('Camera', { model: selected, server, mode })
  }

  const getStep = (name) => {
    const key = Object.keys(STEP_META).find(k => name.includes(k))
    return { ...(STEP_META[key] ?? { color: '#888', label: '', desc: '', badge: '?' }), key }
  }

  const selectedMeta = selected ? getStep(selected) : null

  return (
    <SafeAreaView style={[s.root, { backgroundColor: t.bg }]}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* ── HEADER ── */}
      <View style={s.header}>
        <View style={s.logoRow}>
          <View style={[s.logoDot, { backgroundColor: t.accent }]} />
          <Text style={[s.logoText, { color: t.textPrimary }]}>Tomato</Text>
          <View style={[s.vPill, { backgroundColor: t.accentDim }]}>
            <Text style={[s.vPillText, { color: t.accent }]}>v2.0</Text>
          </View>

          <View style={s.headerActions}>
            {/* History */}
            <TouchableOpacity
              style={[s.iconBtn, { backgroundColor: t.card, borderColor: t.border }]}
              onPress={() => navigation.navigate('History')}
            >
              <Text style={{ fontSize: 13, color: t.textSec }}>☰</Text>
              {history.length > 0 && (
                <View style={[s.historyBadge, { backgroundColor: t.accent }]}>
                  <Text style={s.historyBadgeText}>
                    {history.length > 99 ? '99+' : history.length}
                  </Text>
                </View>
              )}
            </TouchableOpacity>

            {/* Settings */}
            <TouchableOpacity
              style={[s.iconBtn, { backgroundColor: t.card, borderColor: t.border }]}
              onPress={() => navigation.navigate('Settings')}
            >
              <Text style={{ fontSize: 14, color: t.textSec }}>⚙</Text>
            </TouchableOpacity>

            {/* Theme toggle */}
            <TouchableOpacity
              onPress={handleThemeToggle}
              activeOpacity={0.85}
              style={[s.iconBtn, { backgroundColor: t.card, borderColor: t.border }]}
            >
              <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
                <Image
                  source={isDark
                    ? require('../assets/images/red.png')
                    : require('../assets/images/black.png')}
                  style={s.themeIcon}
                />
              </Animated.View>
            </TouchableOpacity>
          </View>
        </View>
        <Text style={[s.tagline, { color: t.textSec }]}>Tomato quality segmentation</Text>
      </View>

      {/* ── LOADING ── */}
      {loading && (
        <View style={s.center}>
          <ActivityIndicator color={t.accent} size="large" />
          <Text style={[s.loadingText, { color: t.textTertiary }]}>Connecting to server...</Text>
          <Text style={[s.serverNote, { color: t.textMuted }]}>{server}</Text>
        </View>
      )}

      {/* ── ERROR ── */}
      {!loading && error && (
        <View style={s.center}>
          <View style={[s.errorCard, { backgroundColor: t.card, borderColor: t.border }]}>
            <View style={s.errorIconWrap}>
              <Text style={s.errorIconText}>⚠</Text>
            </View>
            <Text style={[s.errorTitle, { color: t.textPrimary }]}>Cannot reach server</Text>
            <Text style={[s.errorSub, { color: t.textSec }]}>
              Check your hotspot and server IP address
            </Text>
            <View style={[s.errorUrlBox, { backgroundColor: t.cardAlt, borderColor: t.border }]}>
              <Text style={[s.errorUrl, { color: t.step1 }]}>{server}</Text>
            </View>
            <View style={s.errorActions}>
              <TouchableOpacity
                style={[s.retryBtn, { backgroundColor: t.accent }]}
                onPress={handleRetry}
                disabled={retrying}
              >
                <Text style={s.retryBtnText}>{retrying ? 'Retrying...' : '↺  Retry'}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[s.settingsBtn, { backgroundColor: t.cardAlt, borderColor: t.border }]}
                onPress={() => navigation.navigate('Settings')}
              >
                <Text style={[s.settingsBtnText, { color: t.textPrimary }]}>⚙  Settings</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}

      {/* ── MAIN CONTENT ── */}
      {!loading && !error && (
        <Animated.View style={{ flex: 1, opacity: fadeAnim }}>

          <Text style={[s.sectionLabel, { color: t.textMuted }]}>SELECT MODEL</Text>

          <FlatList
            data={models}
            keyExtractor={i => i}
            style={s.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => {
              const meta = getStep(item)
              const active = item === selected
              const encoder = item.includes('EfficientNet') ? 'EfficientNet-B0' : 'MobileNetV2'
              return (
                <TouchableOpacity
                  style={[
                    s.modelRow,
                    { backgroundColor: t.card, borderColor: active ? meta.color : t.border },
                    active && { borderWidth: 1.5 },
                  ]}
                  onPress={() => setSelected(item)}
                  activeOpacity={0.75}
                >
                  <View style={[s.stepBadge, { backgroundColor: meta.color + '22' }]}>
                    <Text style={[s.stepBadgeText, { color: meta.color }]}>{meta.badge}</Text>
                  </View>
                  <View style={s.modelInfo}>
                    <Text style={[s.modelName, { color: t.textPrimary }]}>{encoder}</Text>
                    <Text style={[s.modelDesc, { color: t.textTertiary }]}>{meta.desc}</Text>
                  </View>
                  {active && (
                    <View style={[s.activeCheck, { backgroundColor: meta.color }]}>
                      <Text style={s.checkMark}>✓</Text>
                    </View>
                  )}
                </TouchableOpacity>
              )
            }}
          />

          {/* Info bar */}
          {selectedMeta && (
            <View style={[s.infoBar, { borderLeftColor: selectedMeta.color, backgroundColor: selectedMeta.color + '08' }]}>
              <View style={s.infoBarInner}>
                <View style={[s.infoBarDot, { backgroundColor: selectedMeta.color }]} />
                <Text style={[s.infoBarText, { color: t.infoText }]}>
                  {STEP_INFO[selectedMeta.key] ?? ''}
                </Text>
              </View>
            </View>
          )}

          {/* Mode selector */}
          <Text style={[s.sectionLabel, { color: t.textMuted }]}>CHOOSE MODE</Text>
          <View style={[s.modeRow, { backgroundColor: t.card, borderColor: t.border }]}>
            {MODES.map((mode, idx) => (
              <React.Fragment key={mode.key}>
                {idx > 0 && <View style={[s.modeDivider, { backgroundColor: t.border }]} />}
                <TouchableOpacity style={s.modeBtn} onPress={() => go(mode.key)} activeOpacity={0.8}>
                  <View style={[s.modeIconWrap, { backgroundColor: mode.color + '18' }]}>
                    <Text style={[s.modeIconText, { color: mode.color }]}>{mode.icon}</Text>
                  </View>
                  <Text style={[s.modeBtnTitle, { color: t.textPrimary }]}>{mode.label}</Text>
                  <Text style={[s.modeBtnSub, { color: t.textSec }]}>{mode.sub}</Text>
                </TouchableOpacity>
              </React.Fragment>
            ))}
          </View>

        </Animated.View>
      )}
    </SafeAreaView>
  )
}

const makeStyles = (t) => StyleSheet.create({
  root: { flex: 1 },
  header: { paddingHorizontal: 20, paddingTop: 16, paddingBottom: 14 },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 4 },
  logoDot: { width: 10, height: 10, borderRadius: 5 },
  logoText: { fontSize: 22, fontWeight: '700', letterSpacing: -0.5 },
  vPill: { paddingHorizontal: 8, paddingVertical: 2, borderRadius: 20 },
  vPillText: { fontSize: 11, fontWeight: '600' },
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: 8, marginLeft: 'auto' },
  iconBtn: {
    width: 34, height: 34, borderRadius: 10,
    borderWidth: 1, alignItems: 'center', justifyContent: 'center',
    position: 'relative',
  },
  historyBadge: {
    position: 'absolute', top: -5, right: -5,
    minWidth: 16, height: 16, borderRadius: 8,
    alignItems: 'center', justifyContent: 'center',
    paddingHorizontal: 3,
  },
  historyBadgeText: { color: '#fff', fontSize: 9, fontWeight: '800' },
  themeIcon: { width: 18, height: 18 },
  tagline: { fontSize: 12, letterSpacing: 0.3 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  loadingText: { marginTop: 16, fontSize: 14, fontWeight: '500' },
  serverNote: { marginTop: 6, fontSize: 12, fontFamily: 'monospace' },
  errorCard: { borderRadius: 18, borderWidth: 1, padding: 24, alignItems: 'center', width: '100%', gap: 12 },
  errorIconWrap: { width: 56, height: 56, borderRadius: 16, backgroundColor: '#EF9F2718', alignItems: 'center', justifyContent: 'center', marginBottom: 4 },
  errorIconText: { fontSize: 26 },
  errorTitle: { fontSize: 18, fontWeight: '700', letterSpacing: -0.3 },
  errorSub: { fontSize: 13, textAlign: 'center', lineHeight: 20 },
  errorUrlBox: { paddingHorizontal: 14, paddingVertical: 9, borderRadius: 8, borderWidth: 1, width: '100%' },
  errorUrl: { fontSize: 12, fontFamily: 'monospace', textAlign: 'center' },
  errorActions: { flexDirection: 'row', gap: 10, width: '100%', marginTop: 4 },
  retryBtn: { flex: 1, paddingVertical: 13, borderRadius: 11, alignItems: 'center' },
  retryBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  settingsBtn: { flex: 1, paddingVertical: 13, borderRadius: 11, borderWidth: 1, alignItems: 'center' },
  settingsBtnText: { fontWeight: '600', fontSize: 14 },
  sectionLabel: { fontSize: 10, fontWeight: '700', letterSpacing: 1.3, paddingHorizontal: 20, marginTop: 6, marginBottom: 8 },
  list: { paddingHorizontal: 16, maxHeight: 270 },
  modelRow: { flexDirection: 'row', alignItems: 'center', borderRadius: 13, padding: 13, marginBottom: 8, borderWidth: 1 },
  stepBadge: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8, marginRight: 12, minWidth: 36, alignItems: 'center' },
  stepBadgeText: { fontSize: 11, fontWeight: '800', letterSpacing: 0.5 },
  modelInfo: { flex: 1 },
  modelName: { fontSize: 14, fontWeight: '600' },
  modelDesc: { fontSize: 12, marginTop: 2 },
  activeCheck: { width: 24, height: 24, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  checkMark: { color: '#fff', fontSize: 12, fontWeight: '800' },
  infoBar: { marginHorizontal: 16, marginBottom: 16, paddingLeft: 14, paddingVertical: 11, paddingRight: 14, borderLeftWidth: 3, borderRadius: 8, marginTop: 2 },
  infoBarInner: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  infoBarDot: { width: 6, height: 6, borderRadius: 3, marginTop: 4, flexShrink: 0 },
  infoBarText: { fontSize: 12, lineHeight: 18, flex: 1 },
  modeRow: { flexDirection: 'row', marginHorizontal: 16, marginBottom: 24, borderRadius: 16, borderWidth: 1, overflow: 'hidden' },
  modeBtn: { flex: 1, alignItems: 'center', paddingVertical: 20, paddingHorizontal: 4 },
  modeDivider: { width: 1, marginVertical: 16 },
  modeIconWrap: { width: 46, height: 46, borderRadius: 23, alignItems: 'center', justifyContent: 'center', marginBottom: 8 },
  modeIconText: { fontSize: 20 },
  modeBtnTitle: { fontSize: 14, fontWeight: '700', letterSpacing: -0.2 },
  modeBtnSub: { fontSize: 11, marginTop: 2 },
})