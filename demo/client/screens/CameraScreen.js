import React, { useState, useEffect, useRef, useCallback } from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { CameraView, useCameraPermissions } from 'expo-camera'
import * as ImagePicker from 'expo-image-picker'
import CoverageBars from '../components/CoverageBars'
import ExportCard from '../components/ExportCard'
import { useTheme } from '../context/ThemeContext'
import { useScanHistory } from '../context/ScanHistoryContext'
import {
  View, Text, Image, TouchableOpacity, StyleSheet,
  Dimensions, Switch, StatusBar, Animated, ScrollView,
} from 'react-native'

const { width: W } = Dimensions.get('window')
const INTERVAL_MS = 1500

const CLASS_LEGEND = [
  { label: 'Fully Ripened (Big)', color: '#FF5078' },
  { label: 'Half Ripened (Big)', color: '#5078FF' },
  { label: 'Green (Big)', color: '#50FFB4' },
  { label: 'Fully Ripened (Little)', color: '#FF8C50' },
  { label: 'Half Ripened (Little)', color: '#7850FF' },
  { label: 'Green (Little)', color: '#50C850' },
]

export default function CameraScreen({ route, navigation }) {
  const { model, server, mode } = route.params
  const { theme: t } = useTheme()
  const { addScan } = useScanHistory()
  const s = makeStyles(t)

  const cameraRef = useRef(null)
  const intervalRef = useRef(null)
  const processingRef = useRef(false)
  const pulseAnim = useRef(new Animated.Value(1)).current
  const scanAnim = useRef(new Animated.Value(0)).current
  const overlayAnim = useRef(new Animated.Value(0)).current
  const analyzingFade = useRef(new Animated.Value(0)).current

  const [photoMode, setPhotoMode] = useState(mode === 'photo')
  const [uploadMode, setUploadMode] = useState(mode === 'upload')
  const isStillMode = photoMode || uploadMode

  const [permission, requestPermission] = useCameraPermissions()
  const [overlay, setOverlay] = useState(null)
  const [coverage, setCoverage] = useState({})
  const [confidence, setConfidence] = useState({})
  const [running, setRunning] = useState(false)
  const [latency, setLatency] = useState(null)
  const [inferenceMs, setInferenceMs] = useState(null)
  const [facing, setFacing] = useState('back')
  const [showOverlay, setShowOverlay] = useState(true)
  const [showLegend, setShowLegend] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [capturedURI, setCapturedURI] = useState(null)

  useEffect(() => {
    if (!permission?.granted) requestPermission()
  }, [permission?.granted, requestPermission])

  useEffect(() => () => stopInference(), [])

  // Scan line animation
  useEffect(() => {
    if (running) {
      const loop = Animated.loop(
        Animated.sequence([
          Animated.timing(scanAnim, { toValue: 1, duration: 1800, useNativeDriver: true }),
          Animated.timing(scanAnim, { toValue: 0, duration: 0, useNativeDriver: true }),
        ])
      )
      loop.start()
      return () => loop.stop()
    } else {
      scanAnim.setValue(0)
    }
  }, [running, scanAnim])

  // Overlay fade
  useEffect(() => {
    if (overlay) {
      Animated.timing(overlayAnim, { toValue: 1, duration: 250, useNativeDriver: true }).start()
    } else {
      overlayAnim.setValue(0)
    }
  }, [overlay, overlayAnim])

  // Analyzing fade
  useEffect(() => {
    if (analyzing) {
      Animated.timing(analyzingFade, { toValue: 1, duration: 200, useNativeDriver: true }).start()
    } else {
      Animated.timing(analyzingFade, { toValue: 0, duration: 150, useNativeDriver: true }).start()
    }
  }, [analyzing, analyzingFade])

  const pulse = useCallback(() => {
    Animated.sequence([
      Animated.timing(pulseAnim, { toValue: 1.4, duration: 150, useNativeDriver: true }),
      Animated.timing(pulseAnim, { toValue: 1, duration: 180, useNativeDriver: true }),
    ]).start()
  }, [pulseAnim])

  const sendFrame = useCallback(async (uri = null) => {
    if (processingRef.current) return
    if (!uri && !cameraRef.current) return
    processingRef.current = true
    const t0 = Date.now()
    try {
      let photoUri = uri
      if (!photoUri) {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.5, base64: false, skipProcessing: true,
          shutterSound: false, imageType: 'jpg',
        })
        photoUri = photo.uri
      }
      const form = new FormData()
      form.append('file', { uri: photoUri, name: 'frame.jpg', type: 'image/jpeg' })
      const res = await fetch(
        `${server}/segment?model_name=${encodeURIComponent(model)}`,
        { method: 'POST', body: form }
      )
      const data = await res.json()
      const elapsed = Date.now() - t0

      setOverlay(data.overlay_b64)
      setCoverage(data.coverage ?? {})
      setConfidence(data.confidence ?? {})
      setLatency(elapsed)
      setInferenceMs(data.inference_ms ?? null)
      pulse()

      // Save to history for still mode only (not every live frame)
      if (photoMode || uploadMode) {
        addScan({
          overlayB64: data.overlay_b64,
          coverage: data.coverage ?? {},
          confidence: data.confidence ?? {},
          model,
          latency: elapsed,
          inferenceMs: data.inference_ms ?? null,
          mode: uploadMode ? 'upload' : 'photo',
        })
      }
    } catch (e) {
      console.warn('sendFrame error:', e.message)
    } finally {
      processingRef.current = false
      if (photoMode || uploadMode) setAnalyzing(false)
    }
  }, [model, server, photoMode, uploadMode, pulse, addScan])

  const startInference = () => {
    setRunning(true)
    intervalRef.current = setInterval(sendFrame, INTERVAL_MS)
  }

  const stopInference = () => {
    setRunning(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  const capturePhoto = async () => {
    if (!cameraRef.current || analyzing) return
    setAnalyzing(true)
    setOverlay(null)
    setCoverage({})
    setConfidence({})
    const photo = await cameraRef.current.takePictureAsync({
      quality: 0.85, base64: false, shutterSound: false, imageType: 'jpg',
    })
    setCapturedURI(photo.uri)
    await sendFrame(photo.uri)
  }

  const resetPhoto = () => {
    setCapturedURI(null)
    setOverlay(null)
    setCoverage({})
    setConfidence({})
    setLatency(null)
    setInferenceMs(null)
  }

  useEffect(() => {
    if (mode === 'upload') pickImage()
  }, [mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const pickImage = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync()
    if (!perm.granted) return
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'], quality: 0.85, allowsEditing: false,
    })
    if (result.canceled || !result.assets?.[0]) return
    setAnalyzing(true)
    setOverlay(null)
    setCoverage({})
    setConfidence({})
    setCapturedURI(result.assets[0].uri)
    await sendFrame(result.assets[0].uri)
  }

  const switchMode = (newPhotoMode, newUploadMode) => {
    stopInference()
    setPhotoMode(newPhotoMode)
    setUploadMode(newUploadMode)
    resetPhoto()
  }

  const encoderLabel = model.includes('EfficientNet') ? 'EffNet-B0' : 'MobileV2'
  const stepColor = model.includes('Step1') ? t.step1 : model.includes('Step2') ? t.step2 : t.step3
  const stepLabel = model.includes('Step1') ? 'S1' : model.includes('Step2') ? 'S2' : 'S3'

  const hasResult = overlay && Object.keys(coverage).length > 0 && !analyzing

  // ── PERMISSION SCREEN ────────────────────────────────────────────
  if (!permission?.granted) {
    return (
      <SafeAreaView style={[s.root, { backgroundColor: t.bg }]}>
        <View style={s.permWrap}>
          <View style={[s.permCard, { backgroundColor: t.card, borderColor: t.border }]}>
            <Text style={s.permEmoji}>📷</Text>
            <Text style={[s.permTitle, { color: t.textPrimary }]}>Camera Access</Text>
            <Text style={[s.permSub, { color: t.textSec }]}>
              Camera permission is required to capture and analyze tomato images.
            </Text>
            <TouchableOpacity style={[s.permBtn, { backgroundColor: t.accent }]} onPress={requestPermission}>
              <Text style={s.permBtnText}>Grant Permission</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    )
  }

  // ── MAIN RENDER ──────────────────────────────────────────────────
  return (
    <SafeAreaView style={[s.root, { backgroundColor: t.bg }]}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* ── HUD ── */}
      <View style={[s.hud, { backgroundColor: t.hudBg }]}>
        <TouchableOpacity
          onPress={() => { stopInference(); navigation.goBack() }}
          style={s.hudBack}
        >
          <Text style={[s.hudBackArrow, { color: t.textPrimary }]}>←</Text>
        </TouchableOpacity>

        <View style={s.hudCenter}>
          <View style={[s.stepPill, { backgroundColor: stepColor + '28' }]}>
            <Text style={[s.stepPillText, { color: stepColor }]}>{stepLabel}</Text>
          </View>
          <Text style={[s.hudModel, { color: t.textPrimary }]}>{encoderLabel}</Text>
        </View>

        <View style={s.hudRight}>
          {latency !== null && (
            <Animated.View style={[s.latencyBadge, { transform: [{ scale: pulseAnim }] }]}>
              <Text style={[s.latencyText, { color: t.accent }]}>{latency}ms</Text>
            </Animated.View>
          )}
          {!isStillMode && (
            <TouchableOpacity
              onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
              style={[s.hudBtn, { backgroundColor: t.card + 'aa', borderColor: t.borderLight }]}
            >
              <Text style={[s.hudBtnText, { color: t.textPrimary }]}>⟳</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity
            onPress={() => setShowLegend(v => !v)}
            style={[s.hudBtn, {
              backgroundColor: showLegend ? t.accentDim : t.card + 'aa',
              borderColor: showLegend ? t.accentBorder : t.borderLight,
            }]}
          >
            <Text style={{ fontSize: 13, color: showLegend ? t.accent : t.textSec }}>⬡</Text>
          </TouchableOpacity>
          {/* History shortcut */}
          <TouchableOpacity
            onPress={() => navigation.navigate('History')}
            style={[s.hudBtn, { backgroundColor: t.card + 'aa', borderColor: t.borderLight }]}
          >
            <Text style={{ fontSize: 13, color: t.textSec }}>☰</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* ── VIEWPORT ── */}
      <View style={s.viewport}>
        {(!capturedURI || !isStillMode) && (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing={facing} />
        )}
        {capturedURI && isStillMode && (
          <Image source={{ uri: capturedURI }} style={StyleSheet.absoluteFill} resizeMode="cover" />
        )}
        {overlay && showOverlay && (
          <Animated.Image
            source={{ uri: `data:image/jpeg;base64,${overlay}` }}
            style={[
              StyleSheet.absoluteFill,
              {
                opacity: overlayAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [0, isStillMode ? 1 : 0.65],
                }),
              },
            ]}
            resizeMode="cover"
          />
        )}

        {/* Animated scan line */}
        {running && !isStillMode && (
          <Animated.View
            style={[s.scanLine, {
              transform: [{
                translateY: scanAnim.interpolate({ inputRange: [0, 1], outputRange: [0, W] }),
              }],
            }]}
            pointerEvents="none"
          />
        )}

        {/* Grid */}
        {running && !isStillMode && (
          <View style={s.scanGrid} pointerEvents="none">
            <View style={[s.gridLine, s.gridH, { top: '33%', backgroundColor: t.scanLine }]} />
            <View style={[s.gridLine, s.gridH, { top: '66%', backgroundColor: t.scanLine }]} />
            <View style={[s.gridLine, s.gridV, { left: '33%', backgroundColor: t.scanLine }]} />
            <View style={[s.gridLine, s.gridV, { left: '66%', backgroundColor: t.scanLine }]} />
          </View>
        )}

        {/* Corners */}
        {(!isStillMode || !capturedURI) && (
          <>
            <View style={[s.corner, s.cornerTL, { borderColor: t.accent }]} />
            <View style={[s.corner, s.cornerTR, { borderColor: t.accent }]} />
            <View style={[s.corner, s.cornerBL, { borderColor: t.accent }]} />
            <View style={[s.corner, s.cornerBR, { borderColor: t.accent }]} />
          </>
        )}

        {/* Analyzing overlay */}
        {analyzing && (
          <Animated.View style={[s.analyzingOverlay, { backgroundColor: t.analyzingBg, opacity: analyzingFade }]}>
            <View style={[s.analyzingCard, { backgroundColor: t.analyzingCard }]}>
              <View style={[s.analyzingDot, { backgroundColor: t.accent }]} />
              <Text style={[s.analyzingText, { color: t.textPrimary }]}>Analysing...</Text>
            </View>
          </Animated.View>
        )}

        {/* Inference time badge */}
        {inferenceMs !== null && !analyzing && isStillMode && (
          <View style={[s.inferenceBadge, { backgroundColor: t.hudBg }]}>
            <Text style={[s.inferenceBadgeText, { color: t.textSec }]}>⚡ {inferenceMs}ms inference</Text>
          </View>
        )}

        {/* Legend */}
        {showLegend && (
          <View style={[s.legend, { backgroundColor: t.hudBg, borderColor: t.border }]}>
            <Text style={[s.legendTitle, { color: t.textMuted }]}>CLASS LEGEND</Text>
            {CLASS_LEGEND.map(c => (
              <View key={c.label} style={s.legendRow}>
                <View style={[s.legendDot, { backgroundColor: c.color }]} />
                <Text style={[s.legendLabel, { color: t.textSec }]}>{c.label}</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* ── BOTTOM PANEL ── */}
      <View style={[s.bottomPanel, { backgroundColor: t.bgSecondary }]}>

        {/* Mode tabs */}
        <View style={[s.modeToggleRow, { backgroundColor: t.card, borderColor: t.border }]}>
          {[
            { label: 'Live', pm: false, um: false },
            { label: 'Photo', pm: true, um: false },
            { label: 'Upload', pm: false, um: true },
          ].map(({ label, pm, um }) => {
            const active = pm === photoMode && um === uploadMode
            return (
              <TouchableOpacity
                key={label}
                style={[s.modeTab, active && { backgroundColor: t.accent }]}
                onPress={() => switchMode(pm, um)}
              >
                <Text style={[s.modeTabText, { color: active ? '#fff' : t.textTertiary }]}>
                  {label}
                </Text>
              </TouchableOpacity>
            )
          })}
        </View>

        {/* Coverage + Export */}
        {Object.keys(coverage).length > 0 && !analyzing && (
          <ScrollView
            style={s.coverageScroll}
            showsVerticalScrollIndicator={false}
            contentContainerStyle={{ paddingBottom: 4 }}
          >
            <CoverageBars coverage={coverage} confidence={confidence} />

            {/* Export button — only for still/upload results */}
            {isStillMode && hasResult && (
              <View style={s.exportWrap}>
                <ExportCard
                  overlayB64={overlay}
                  coverage={coverage}
                  confidence={confidence}
                  model={model}
                  latency={latency}
                  inferenceMs={inferenceMs}
                />
              </View>
            )}
          </ScrollView>
        )}

        {/* Controls */}
        <View style={s.controls}>

          {/* LIVE */}
          {!photoMode && !uploadMode && (
            <>
              <TouchableOpacity
                style={[s.mainBtn, running ? s.stopBtn : { backgroundColor: t.accent }]}
                onPress={() => running ? stopInference() : startInference()}
                activeOpacity={0.85}
              >
                <Text style={s.mainBtnText}>{running ? '■  Stop' : '▶  Start'}</Text>
              </TouchableOpacity>
              <View style={s.overlayToggle}>
                <Text style={[s.overlayLabel, { color: t.textTertiary }]}>Overlay</Text>
                <Switch
                  value={showOverlay}
                  onValueChange={setShowOverlay}
                  trackColor={{ true: t.accent, false: t.switchTrackOff }}
                  thumbColor="#fff"
                  ios_backgroundColor={t.switchTrackOff}
                />
              </View>
            </>
          )}

          {/* UPLOAD */}
          {uploadMode && (
            !capturedURI ? (
              <TouchableOpacity
                style={[s.mainBtn, { backgroundColor: t.accent }]}
                onPress={pickImage}
                disabled={analyzing}
                activeOpacity={0.85}
              >
                <Text style={s.mainBtnText}>↑  Choose Image</Text>
              </TouchableOpacity>
            ) : (
              <View style={s.stillActions}>
                <TouchableOpacity
                  style={[s.retakeBtn, { backgroundColor: t.cardAlt, borderColor: t.border }]}
                  onPress={() => { resetPhoto(); pickImage() }}
                >
                  <Text style={[s.retakeBtnText, { color: t.textPrimary }]}>↑  Pick Another</Text>
                </TouchableOpacity>
                <View style={s.overlayToggle}>
                  <Text style={[s.overlayLabel, { color: t.textTertiary }]}>Overlay</Text>
                  <Switch
                    value={showOverlay}
                    onValueChange={setShowOverlay}
                    trackColor={{ true: t.accent, false: t.switchTrackOff }}
                    thumbColor="#fff"
                    ios_backgroundColor={t.switchTrackOff}
                  />
                </View>
              </View>
            )
          )}

          {/* PHOTO */}
          {photoMode && (
            !capturedURI ? (
              <TouchableOpacity
                style={s.captureBtn}
                onPress={capturePhoto}
                disabled={analyzing}
                activeOpacity={0.9}
              >
                <View style={[s.captureRing, { borderColor: t.textPrimary }]}>
                  <View style={[s.captureInner, { backgroundColor: t.textPrimary }]} />
                </View>
              </TouchableOpacity>
            ) : (
              <View style={s.stillActions}>
                <TouchableOpacity
                  style={[s.retakeBtn, { backgroundColor: t.cardAlt, borderColor: t.border }]}
                  onPress={resetPhoto}
                >
                  <Text style={[s.retakeBtnText, { color: t.textPrimary }]}>↩  Retake</Text>
                </TouchableOpacity>
                <View style={s.overlayToggle}>
                  <Text style={[s.overlayLabel, { color: t.textTertiary }]}>Overlay</Text>
                  <Switch
                    value={showOverlay}
                    onValueChange={setShowOverlay}
                    trackColor={{ true: t.accent, false: t.switchTrackOff }}
                    thumbColor="#fff"
                    ios_backgroundColor={t.switchTrackOff}
                  />
                </View>
              </View>
            )
          )}

        </View>
      </View>
    </SafeAreaView>
  )
}

const CORNER_SIZE = 20
const CORNER_W = 2.5

const makeStyles = (t) => StyleSheet.create({
  root: { flex: 1 },
  hud: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  hudBack: { padding: 6, marginRight: 4 },
  hudBackArrow: { fontSize: 22 },
  hudCenter: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  stepPill: { paddingHorizontal: 9, paddingVertical: 4, borderRadius: 8 },
  stepPillText: { fontSize: 11, fontWeight: '800', letterSpacing: 0.5 },
  hudModel: { fontSize: 13, fontWeight: '600' },
  hudRight: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  latencyBadge: {
    backgroundColor: 'rgba(29,158,117,0.2)',
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderRadius: 8,
  },
  latencyText: { fontSize: 11, fontWeight: '700' },
  hudBtn: {
    width: 34, height: 34, borderRadius: 10,
    borderWidth: 1, alignItems: 'center', justifyContent: 'center',
  },
  hudBtnText: { fontSize: 18 },
  viewport: { width: W, height: W, overflow: 'hidden', position: 'relative' },
  scanLine: {
    position: 'absolute', left: 0, right: 0, height: 2,
    backgroundColor: 'rgba(29,158,117,0.7)',
    shadowColor: '#1D9E75', shadowOpacity: 0.8, shadowRadius: 4,
  },
  scanGrid: { ...StyleSheet.absoluteFillObject },
  gridLine: { position: 'absolute' },
  gridH: { left: 0, right: 0, height: 1 },
  gridV: { top: 0, bottom: 0, width: 1 },
  corner: { position: 'absolute', width: CORNER_SIZE, height: CORNER_SIZE },
  cornerTL: { top: 40, left: 40, borderTopWidth: CORNER_W, borderLeftWidth: CORNER_W },
  cornerTR: { top: 40, right: 40, borderTopWidth: CORNER_W, borderRightWidth: CORNER_W },
  cornerBL: { bottom: 40, left: 40, borderBottomWidth: CORNER_W, borderLeftWidth: CORNER_W },
  cornerBR: { bottom: 40, right: 40, borderBottomWidth: CORNER_W, borderRightWidth: CORNER_W },
  analyzingOverlay: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
  analyzingCard: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    borderRadius: 14, paddingHorizontal: 22, paddingVertical: 14,
    shadowOpacity: 0.15, shadowRadius: 10, elevation: 6,
  },
  analyzingDot: { width: 8, height: 8, borderRadius: 4 },
  analyzingText: { fontSize: 14, fontWeight: '600' },
  inferenceBadge: {
    position: 'absolute', bottom: 10, alignSelf: 'center',
    paddingHorizontal: 12, paddingVertical: 5, borderRadius: 20,
  },
  inferenceBadgeText: { fontSize: 11, fontWeight: '500' },
  legend: {
    position: 'absolute', top: 10, right: 10,
    borderRadius: 12, borderWidth: 1, padding: 12, gap: 6, minWidth: 175,
  },
  legendTitle: { fontSize: 9, fontWeight: '700', letterSpacing: 1.2, marginBottom: 2 },
  legendRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  legendDot: { width: 10, height: 10, borderRadius: 5 },
  legendLabel: { fontSize: 11 },
  bottomPanel: { flex: 1, paddingTop: 4 },
  modeToggleRow: {
    flexDirection: 'row', marginHorizontal: 16, marginVertical: 10,
    borderRadius: 11, borderWidth: 1, padding: 3,
  },
  modeTab: { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 8 },
  modeTabText: { fontSize: 13, fontWeight: '600' },
  coverageScroll: { flex: 1 },
  exportWrap: { paddingHorizontal: 16, paddingBottom: 8 },
  controls: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 12, gap: 12,
  },
  mainBtn: {
    flex: 1, paddingVertical: 14, borderRadius: 13,
    alignItems: 'center', justifyContent: 'center',
  },
  stopBtn: { backgroundColor: '#A32D2D' },
  mainBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  overlayToggle: { alignItems: 'center', gap: 3 },
  overlayLabel: { fontSize: 10, fontWeight: '500' },
  captureBtn: { flex: 1, alignItems: 'center' },
  captureRing: {
    width: 68, height: 68, borderRadius: 34,
    borderWidth: 3, alignItems: 'center', justifyContent: 'center',
  },
  captureInner: { width: 54, height: 54, borderRadius: 27 },
  stillActions: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 12 },
  retakeBtn: {
    flex: 1, paddingVertical: 14, borderRadius: 13,
    borderWidth: 1, alignItems: 'center',
  },
  retakeBtnText: { fontSize: 14, fontWeight: '600' },
  permWrap: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  permCard: {
    borderRadius: 18, borderWidth: 1, padding: 28,
    alignItems: 'center', gap: 14, width: '100%',
  },
  permEmoji: { fontSize: 44 },
  permTitle: { fontSize: 20, fontWeight: '700' },
  permSub: { fontSize: 14, textAlign: 'center', lineHeight: 21 },
  permBtn: { paddingVertical: 14, paddingHorizontal: 32, borderRadius: 13, alignItems: 'center', marginTop: 4 },
  permBtnText: { color: '#fff', fontSize: 15, fontWeight: '700' },
})