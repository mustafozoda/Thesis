import React, { useState, useEffect, useRef } from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import {
  View, Text, TouchableOpacity, FlatList,
  StyleSheet, ActivityIndicator,
  StatusBar, Animated, Image
} from 'react-native'
import { useTheme } from '../context/ThemeContext'

const SERVER = 'http://192.168.137.1:8000'

const STEP_META = {
  'Step1': { color: '#378ADD', label: 'Baseline', desc: 'Natural background' },
  'Step2': { color: '#1D9E75', label: 'Best', desc: 'Background removed' },
  'Step3': { color: '#EF9F27', label: 'Synthetic', desc: 'Synthetic background' },
}



export { SERVER }

export default function MenuScreen({ navigation }) {
  const { theme: t, isDark, toggle } = useTheme()
  const s = makeStyles(t)

  const [models, setModels] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const fadeAnim = React.useRef(new Animated.Value(0)).current

  const scaleAnim = useRef(new Animated.Value(1)).current

  const handleToggle = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, { toValue: 0.6, duration: 120, useNativeDriver: true }),
      Animated.timing(scaleAnim, { toValue: 1.2, duration: 150, useNativeDriver: true }),
      Animated.timing(scaleAnim, { toValue: 1.0, duration: 100, useNativeDriver: true }),
    ]).start()
    toggle()
  }

  useEffect(() => {
    fetch(`${SERVER}/models`)
      .then(r => r.json())
      .then(d => {
        setModels(d.models)
        setSelected(d.models[1] ?? d.models[0])
        setLoading(false)
        Animated.timing(fadeAnim, { toValue: 1, duration: 400, useNativeDriver: true }).start()
      })
      .catch(() => { setError(true); setLoading(false) })
  }, [fadeAnim])

  const getStep = (name) => {
    const key = Object.keys(STEP_META).find(k => name.includes(k))
    return STEP_META[key] ?? { color: '#888', label: '', desc: '' }
  }

  const go = (mode) => {
    if (!selected) return
    navigation.navigate('Camera', { model: selected, server: SERVER, mode })
  }

  const selectedMeta = selected ? getStep(selected) : null

  return (
    <SafeAreaView style={s.root}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* Header */}
      <View style={s.header}>
        <View style={s.logoRow}>
          <View style={s.logoDot} />
          <Text style={s.logoText}>Tomato</Text>
          <View style={[s.pill, { backgroundColor: '#1D9E7522' }]}>
            <Text style={[s.pillText, { color: '#1D9E75' }]}>v1.0</Text>
          </View>
          <TouchableOpacity onPress={handleToggle} style={s.themeToggle} activeOpacity={1}>
            <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
              <Image
                source={isDark ? require('../assets/images/red.png') : require('../assets/images/black.png')}
                style={s.themeToggleIcon}
              />
            </Animated.View>
          </TouchableOpacity>
        </View>
        <Text style={s.tagline}>Tomato quality segmentation</Text>
      </View>

      {loading && (
        <View style={s.center}>
          <ActivityIndicator color="#1D9E75" size="large" />
          <Text style={s.loadingText}>Connecting to server...</Text>
        </View>
      )}

      {error && (
        <View style={s.center}>
          <View style={s.errorCard}>
            <Text style={s.errorIcon}>⚠</Text>
            <Text style={s.errorTitle}>Cannot reach server</Text>
            <Text style={s.errorSub}>Check your hotspot and IP address</Text>
            <Text style={s.errorIP}>{SERVER}</Text>
          </View>
        </View>
      )}

      {!loading && !error && (
        <Animated.View style={{ flex: 1, opacity: fadeAnim }}>

          <Text style={s.sectionLabel}>SELECT MODEL</Text>
          <FlatList
            data={models}
            keyExtractor={i => i}
            style={s.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => {
              const meta = getStep(item)
              const active = item === selected
              const encoder = item.includes('EfficientNet') ? 'EffNet-B0' : 'MobileNetV2'
              return (
                <TouchableOpacity
                  style={[s.modelRow, active && { borderColor: meta.color, borderWidth: 1.5 }]}
                  onPress={() => setSelected(item)}
                  activeOpacity={0.75}
                >
                  <View style={[s.stepBadge, { backgroundColor: meta.color + '22' }]}>
                    <Text style={[s.stepBadgeText, { color: meta.color }]}>{meta.label}</Text>
                  </View>
                  <View style={s.modelInfo}>
                    <Text style={s.modelName}>{encoder}</Text>
                    <Text style={s.modelDesc}>{meta.desc}</Text>
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

          {selectedMeta && (
            <View style={[s.infoBar, { borderLeftColor: selectedMeta.color }]}>
              <Text style={s.infoBarText}>
                {selected?.includes('Step2')
                  ? 'Best performing model — mIoU 0.76'
                  : selected?.includes('Step3')
                    ? 'Trained on synthetic backgrounds'
                    : 'Natural background baseline'}
              </Text>
            </View>
          )}

          <Text style={s.sectionLabel}>CHOOSE MODE</Text>
          <View style={s.modeRow}>
            <TouchableOpacity style={s.modeBtn} onPress={() => go('live')} activeOpacity={0.8}>
              <View style={[s.modeIcon, { backgroundColor: '#1D9E7520' }]}>
                <Text style={[s.modeIconText, { color: '#1D9E75' }]}>◉</Text>
              </View>
              <Text style={s.modeBtnTitle}>Live</Text>
              <Text style={s.modeBtnSub}>Real-time</Text>
            </TouchableOpacity>

            <View style={s.modeDivider} />

            <TouchableOpacity style={s.modeBtn} onPress={() => go('photo')} activeOpacity={0.8}>
              <View style={[s.modeIcon, { backgroundColor: '#378ADD20' }]}>
                <Text style={[s.modeIconText, { color: '#378ADD' }]}>⬡</Text>
              </View>
              <Text style={s.modeBtnTitle}>Photo</Text>
              <Text style={s.modeBtnSub}>Capture</Text>
            </TouchableOpacity>

            <View style={s.modeDivider} />

            <TouchableOpacity style={s.modeBtn} onPress={() => go('upload')} activeOpacity={0.8}>
              <View style={[s.modeIcon, { backgroundColor: '#EF9F2720' }]}>
                <Text style={[s.modeIconText, { color: '#EF9F27' }]}>↑</Text>
              </View>
              <Text style={s.modeBtnTitle}>Upload</Text>
              <Text style={s.modeBtnSub}>From gallery</Text>
            </TouchableOpacity>
          </View>

        </Animated.View>
      )}
    </SafeAreaView>
  )
}

const makeStyles = (t) => StyleSheet.create({
  root: { flex: 1, backgroundColor: t.bg },
  header: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 16 },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 4 },
  logoDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#1D9E75' },
  logoText: { fontSize: 22, fontWeight: '700', color: t.textPrimary, letterSpacing: -0.5 },
  pill: { paddingHorizontal: 8, paddingVertical: 2, borderRadius: 20 },
  pillText: { fontSize: 11, fontWeight: '600' },
  themeToggle: { marginLeft: 'auto', padding: 4 },
  themeToggleIcon: { width: 24, height: 24 },
  // themeToggleIcon: { width: 24, height: 24, tintColor: t.textPrimary },
  themeToggleText: { fontSize: 18 },
  tagline: { fontSize: 13, color: t.textSec, letterSpacing: 0.3 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  loadingText: { color: t.textTertiary, marginTop: 16, fontSize: 14 },
  errorCard: { backgroundColor: t.cardAlt, borderRadius: 16, padding: 24, alignItems: 'center', width: '100%' },
  errorIcon: { fontSize: 32, marginBottom: 12, color: '#EF9F27' },
  errorTitle: { color: t.textPrimary, fontSize: 17, fontWeight: '600', marginBottom: 6 },
  errorSub: { color: t.textSec, fontSize: 13, marginBottom: 12 },
  errorIP: { color: '#378ADD', fontSize: 12, fontFamily: 'monospace' },
  sectionLabel: {
    fontSize: 11, color: t.textMuted, fontWeight: '600', letterSpacing: 1.2,
    paddingHorizontal: 20, marginTop: 8, marginBottom: 8,
  },
  list: { paddingHorizontal: 16, maxHeight: 280 },
  modelRow: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: t.card,
    borderRadius: 12, padding: 12, marginBottom: 8,
    borderWidth: 1, borderColor: t.border,
  },
  stepBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8, marginRight: 12 },
  stepBadgeText: { fontSize: 11, fontWeight: '700' },
  modelInfo: { flex: 1 },
  modelName: { color: t.textPrimary, fontSize: 14, fontWeight: '500' },
  modelDesc: { color: t.textTertiary, fontSize: 12, marginTop: 2 },
  activeCheck: { width: 22, height: 22, borderRadius: 11, alignItems: 'center', justifyContent: 'center' },
  checkMark: { color: '#fff', fontSize: 12, fontWeight: '700' },
  infoBar: {
    marginHorizontal: 16, marginBottom: 16, paddingLeft: 12,
    paddingVertical: 10, borderLeftWidth: 3,
  },
  infoBarText: { color: t.infoText, fontSize: 13 },
  modeRow: {
    flexDirection: 'row', marginHorizontal: 16, marginBottom: 24,
    backgroundColor: t.card, borderRadius: 16,
    borderWidth: 1, borderColor: t.border, overflow: 'hidden',
  },
  modeBtn: { flex: 1, alignItems: 'center', paddingVertical: 20 },
  modeDivider: { width: 1, backgroundColor: t.border, marginVertical: 16 },
  modeIcon: {
    width: 44, height: 44, borderRadius: 22,
    alignItems: 'center', justifyContent: 'center', marginBottom: 8,
  },
  modeIconText: { fontSize: 20 },
  modeBtnTitle: { color: t.textPrimary, fontSize: 15, fontWeight: '600' },
  modeBtnSub: { color: t.textSec, fontSize: 12, marginTop: 2 },
})