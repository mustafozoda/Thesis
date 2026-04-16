import React, { useState, useEffect, useRef, useCallback } from 'react'
import { SafeAreaView } from 'react-native-safe-area-context'
import { CameraView, useCameraPermissions } from 'expo-camera'
import * as ImagePicker from 'expo-image-picker'
import CoverageBars from '../components/CoverageBars'
import { useTheme } from '../context/ThemeContext'
import {
  View, Text, Image, TouchableOpacity, StyleSheet,
  Dimensions, Switch, StatusBar, Animated,
} from 'react-native'

const { width: W } = Dimensions.get('window')
const INTERVAL_MS = 1500

export default function CameraScreen({ route, navigation }) {
  const { model, server, mode } = route.params
  const { theme: t } = useTheme()
  const s = makeStyles(t)

  const cameraRef = useRef(null)
  const intervalRef = useRef(null)
  const processingRef = useRef(false)
  const pulseAnim = useRef(new Animated.Value(1)).current

  const [photoMode, setPhotoMode] = useState(mode === 'photo')
  const [uploadMode, setUploadMode] = useState(mode === 'upload')
  const isStillMode = photoMode || uploadMode

  const [permission, requestPermission] = useCameraPermissions()
  const [overlay, setOverlay] = useState(null)
  const [coverage, setCoverage] = useState({})
  const [running, setRunning] = useState(false)
  const [latency, setLatency] = useState(null)
  const [facing, setFacing] = useState('back')
  const [showOverlay, setShowOverlay] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [capturedURI, setCapturedURI] = useState(null)

  useEffect(() => {
    if (!permission?.granted) requestPermission()
  }, []) // eslint-disable-line

  useEffect(() => () => stopInference(), [])

  const pulse = useCallback(() => {
    Animated.sequence([
      Animated.timing(pulseAnim, { toValue: 1.3, duration: 200, useNativeDriver: true }),
      Animated.timing(pulseAnim, { toValue: 1, duration: 200, useNativeDriver: true }),
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
          quality: 0.5, base64: false, skipProcessing: true, shutterSound: false,
          imageType: 'jpg', pictureSize: `${W}x${W}`,
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
      setOverlay(data.overlay_b64)
      setCoverage(data.coverage ?? {})
      setLatency(Date.now() - t0)
      pulse()
    } catch (e) {
      console.warn(e.message)
    } finally {
      processingRef.current = false
      if (photoMode || uploadMode) setAnalyzing(false)
    }
  }, [model, server, photoMode, uploadMode]) // eslint-disable-line

  const startInference = () => {
    setRunning(true)
    intervalRef.current = setInterval(sendFrame, INTERVAL_MS)
  }

  const stopInference = () => {
    setRunning(false)
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }

  const capturePhoto = async () => {
    if (!cameraRef.current || analyzing) return
    setAnalyzing(true)
    setOverlay(null)
    setCoverage({})
    const photo = await cameraRef.current.takePictureAsync({
      quality: 0.8, base64: false, shutterSound: false,
      imageType: 'jpg', pictureSize: `${W}x${W}`,
    })
    setCapturedURI(photo.uri)
    await sendFrame(photo.uri)
  }

  const resetPhoto = () => {
    setCapturedURI(null)
    setOverlay(null)
    setCoverage({})
    setLatency(null)
  }

  useEffect(() => {
    if (mode === 'upload') pickImage()
  }, []) // eslint-disable-line

  const pickImage = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync()
    if (!perm.granted) return
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'], quality: 0.8, allowsEditing: false,
    })
    if (result.canceled || !result.assets?.[0]) return
    setAnalyzing(true)
    setOverlay(null)
    setCoverage({})
    setCapturedURI(result.assets[0].uri)
    await sendFrame(result.assets[0].uri)
  }

  const encoderShort = model.includes('EfficientNet') ? 'EffNet-B0' : 'MobileV2'
  const stepLabel = model.includes('Step1') ? 'S1' : model.includes('Step2') ? 'S2' : 'S3'
  const stepColor = model.includes('Step1') ? '#378ADD' : model.includes('Step2') ? '#1D9E75' : '#EF9F27'

  if (!permission?.granted) return (
    <SafeAreaView style={s.root}>
      <Text style={s.permText}>Camera permission needed</Text>
      <TouchableOpacity style={s.permBtn} onPress={requestPermission}>
        <Text style={s.permBtnText}>Grant Permission</Text>
      </TouchableOpacity>
    </SafeAreaView>
  )

  const topForeground = (
    <View style={s.hud}>
      <TouchableOpacity onPress={() => { stopInference(); navigation.goBack() }} style={s.backBtn}>
        <Text style={s.backArrow}>←</Text>
      </TouchableOpacity>

      <View style={s.hudCenter}>
        <View style={[s.stepPill, { backgroundColor: stepColor + '33' }]}>
          <Text style={[s.stepPillText, { color: stepColor }]}>{stepLabel}</Text>
        </View>
        <Text style={s.hudModel}>{encoderShort}</Text>
      </View>

      <View style={s.hudRight}>
        {latency && (
          <Animated.View style={[s.latencyBadge, { transform: [{ scale: pulseAnim }] }]}>
            <Text style={s.latencyText}>{latency}ms</Text>
          </Animated.View>
        )}
        <TouchableOpacity
          onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
          style={s.flipBtn}
        >
          <Text style={s.flipText}>⟳</Text>
        </TouchableOpacity>
      </View>
    </View>
  )

  return (
    <SafeAreaView style={s.root}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />
      {topForeground}

      {/* Camera viewport */}
      <View style={s.viewport}>
        {(!capturedURI || !isStillMode) && (
          <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing={facing} />
        )}
        {capturedURI && isStillMode && !showOverlay && (
          <Image source={{ uri: capturedURI }} style={StyleSheet.absoluteFill} resizeMode="cover" />
        )}
        {capturedURI && isStillMode && analyzing && (
          <Image source={{ uri: capturedURI }} style={StyleSheet.absoluteFill} resizeMode="cover" />
        )}
        {overlay && showOverlay && isStillMode && (
          <Image
            source={{ uri: `data:image/jpeg;base64,${overlay}` }}
            style={StyleSheet.absoluteFill} resizeMode="cover"
          />
        )}
        {overlay && showOverlay && !isStillMode && (
          <Image
            source={{ uri: `data:image/jpeg;base64,${overlay}` }}
            style={[StyleSheet.absoluteFill, { opacity: 0.6 }]}
            resizeMode="cover"
          />
        )}

        {/* Scanning grid lines when live */}
        {running && !isStillMode && (
          <View style={s.scanGrid} pointerEvents="none">
            <View style={[s.scanLine, { top: '33%' }]} />
            <View style={[s.scanLine, { top: '66%' }]} />
            <View style={[s.scanLineV, { left: '33%' }]} />
            <View style={[s.scanLineV, { left: '66%' }]} />
          </View>
        )}

        {/* Analyzing overlay */}
        {analyzing && (
          <View style={s.analyzingOverlay}>
            <View style={s.analyzingCard}>
              <Text style={s.analyzingText}>Analysing...</Text>
            </View>
          </View>
        )}

        {/* Corner brackets */}
        {(!isStillMode || !capturedURI) && <>
          <View style={[s.corner, s.cornerTL]} />
          <View style={[s.corner, s.cornerTR]} />
          <View style={[s.corner, s.cornerBL]} />
          <View style={[s.corner, s.cornerBR]} />
        </>}
      </View>

      {/* Bottom panel */}
      <View style={s.bottomPanel}>

        {/* Mode toggle */}
        <View style={s.modeToggleRow}>
          <TouchableOpacity
            style={[s.modeToggleBtn, !photoMode && !uploadMode && s.modeToggleActive]}
            onPress={() => { stopInference(); setPhotoMode(false); setUploadMode(false); resetPhoto() }}
          >
            <Text style={[s.modeToggleText, !photoMode && !uploadMode && s.modeToggleTextActive]}>Live</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[s.modeToggleBtn, photoMode && s.modeToggleActive]}
            onPress={() => { stopInference(); setPhotoMode(true); setUploadMode(false); resetPhoto() }}
          >
            <Text style={[s.modeToggleText, photoMode && s.modeToggleTextActive]}>Photo</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[s.modeToggleBtn, uploadMode && s.modeToggleActive]}
            onPress={() => { stopInference(); setPhotoMode(false); setUploadMode(true); resetPhoto() }}
          >
            <Text style={[s.modeToggleText, uploadMode && s.modeToggleTextActive]}>Upload</Text>
          </TouchableOpacity>
        </View>

        {/* Coverage bars */}
        {coverage && Object.keys(coverage).length > 0 && (
          <View style={s.coverageContainer}>
            <CoverageBars coverage={coverage} />
          </View>
        )}

        {/* Controls */}
        <View style={s.controls}>
          {!photoMode && !uploadMode ? (
            <>
              <TouchableOpacity
                style={[s.mainBtn, running ? s.stopBtn : s.startBtn]}
                onPress={() => running ? stopInference() : startInference()}
              >
                <Text style={s.mainBtnText}>{running ? '■  Stop' : '▶  Start'}</Text>
              </TouchableOpacity>
              <View style={s.overlayToggle}>
                <Text style={s.overlayLabel}>Overlay</Text>
                <Switch
                  value={showOverlay}
                  onValueChange={setShowOverlay}
                  trackColor={{ true: '#1D9E75', false: t.switchTrackOff }}
                  thumbColor="#fff"
                />
              </View>
            </>
          ) : uploadMode ? (
            !capturedURI ? (
              <TouchableOpacity style={s.mainBtn} onPress={pickImage} disabled={analyzing}>
                <Text style={s.mainBtnText}>↑  Choose Image</Text>
              </TouchableOpacity>
            ) : (
              <View style={s.photoActions}>
                <TouchableOpacity style={s.retakeBtn} onPress={() => { resetPhoto(); pickImage() }}>
                  <Text style={s.retakeBtnText}>↑  Pick another</Text>
                </TouchableOpacity>
                <View style={s.overlayToggle}>
                  <Text style={s.overlayLabel}>Overlay</Text>
                  <Switch
                    value={showOverlay}
                    onValueChange={setShowOverlay}
                    trackColor={{ true: '#1D9E75', false: t.switchTrackOff }}
                    thumbColor="#fff"
                  />
                </View>
              </View>
            )
          ) : (
            <>
              {!capturedURI ? (
                <TouchableOpacity style={s.captureBtn} onPress={capturePhoto} disabled={analyzing}>
                  <View style={s.captureRing}>
                    <View style={s.captureInner} />
                  </View>
                </TouchableOpacity>
              ) : (
                <View style={s.photoActions}>
                  <TouchableOpacity style={s.retakeBtn} onPress={resetPhoto}>
                    <Text style={s.retakeBtnText}>↩  Retake</Text>
                  </TouchableOpacity>
                  <View style={s.overlayToggle}>
                    <Text style={s.overlayLabel}>Overlay</Text>
                    <Switch
                      value={showOverlay}
                      onValueChange={setShowOverlay}
                      trackColor={{ true: '#1D9E75', false: t.switchTrackOff }}
                      thumbColor="#fff"
                    />
                  </View>
                </View>
              )}
            </>
          )}
        </View>
      </View>
    </SafeAreaView>
  )
}

const CORNER_SIZE = 18
const CORNER_W = 2

const makeStyles = (t) => StyleSheet.create({
  root: { flex: 1, backgroundColor: t.bg },
  viewport: { width: W, height: W, position: 'relative', overflow: 'hidden' },
  hud: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 12, paddingTop: 10, paddingBottom: 8,
    backgroundColor: t.hudBg,
  },
  backBtn: { padding: 6 },
  backArrow: { color: t.textPrimary, fontSize: 20 },
  hudCenter: {
    flex: 1, flexDirection: 'row', alignItems: 'center',
    justifyContent: 'center', gap: 8,
  },
  stepPill: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8 },
  stepPillText: { fontSize: 11, fontWeight: '700' },
  hudModel: { color: t.textPrimary, fontSize: 13, fontWeight: '500' },
  hudRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  latencyBadge: {
    backgroundColor: 'rgba(29,158,117,0.3)',
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8,
  },
  latencyText: { color: '#1D9E75', fontSize: 11, fontWeight: '600' },
  flipBtn: { padding: 6 },
  flipText: { color: t.textPrimary, fontSize: 20 },
  scanGrid: { ...StyleSheet.absoluteFillObject },
  scanLine: {
    position: 'absolute', left: 0, right: 0, height: 1,
    backgroundColor: t.scanLine,
  },
  scanLineV: {
    position: 'absolute', top: 0, bottom: 0, width: 1,
    backgroundColor: t.scanLine,
  },
  analyzingOverlay: {
    ...StyleSheet.absoluteFillObject, backgroundColor: t.analyzingBg,
    alignItems: 'center', justifyContent: 'center',
  },
  analyzingCard: {
    backgroundColor: t.analyzingCard, borderRadius: 12,
    paddingHorizontal: 24, paddingVertical: 14,
    // shadow for light mode
    shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 8, elevation: 4,
  },
  analyzingText: { color: t.textPrimary, fontSize: 15, fontWeight: '500' },
  corner: { position: 'absolute', width: CORNER_SIZE, height: CORNER_SIZE },
  cornerTL: { top: 50, left: 50, borderTopWidth: CORNER_W, borderLeftWidth: CORNER_W, borderColor: '#1D9E75' },
  cornerTR: { top: 50, right: 50, borderTopWidth: CORNER_W, borderRightWidth: CORNER_W, borderColor: '#1D9E75' },
  cornerBL: { bottom: 50, left: 50, borderBottomWidth: CORNER_W, borderLeftWidth: CORNER_W, borderColor: '#1D9E75' },
  cornerBR: { bottom: 50, right: 50, borderBottomWidth: CORNER_W, borderRightWidth: CORNER_W, borderColor: '#1D9E75' },
  bottomPanel: { flex: 1, backgroundColor: t.bgSecondary, paddingTop: 4 },
  coverageContainer: { flex: 1, justifyContent: 'flex-start' },
  modeToggleRow: {
    flexDirection: 'row', marginHorizontal: 16, marginVertical: 10,
    backgroundColor: t.card, borderRadius: 10,
    borderWidth: 1, borderColor: t.border, padding: 3,
  },
  modeToggleBtn: { flex: 1, paddingVertical: 7, alignItems: 'center', borderRadius: 8 },
  modeToggleActive: { backgroundColor: '#1D9E75' },
  modeToggleText: { color: t.textTertiary, fontSize: 13, fontWeight: '600' },
  modeToggleTextActive: { color: '#fff' },
  controls: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 12, gap: 12,
  },
  mainBtn: {
    flex: 1, paddingVertical: 14, borderRadius: 12,
    alignItems: 'center', justifyContent: 'center',
  },
  startBtn: { backgroundColor: '#1D9E75' },
  stopBtn: { backgroundColor: '#A32D2D' },
  mainBtnText: { color: t.textPrimary, fontSize: 15, fontWeight: '700' },
  overlayToggle: { alignItems: 'center', gap: 2 },
  overlayLabel: { color: t.textTertiary, fontSize: 11 },
  captureBtn: { flex: 1, alignItems: 'center' },
  captureRing: {
    width: 64, height: 64, borderRadius: 32,
    borderWidth: 3, borderColor: t.textPrimary,
    alignItems: 'center', justifyContent: 'center',
  },
  captureInner: { width: 50, height: 50, borderRadius: 25, backgroundColor: t.textPrimary },
  photoActions: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 12 },
  retakeBtn: {
    flex: 1, paddingVertical: 14, borderRadius: 12,
    backgroundColor: t.cardAlt, borderWidth: 1, borderColor: t.border,
    alignItems: 'center',
  },
  retakeBtnText: { color: t.textPrimary, fontSize: 15, fontWeight: '600' },
  permText: { color: t.textPrimary, textAlign: 'center', marginTop: 100, fontSize: 16 },
  permBtn: {
    margin: 24, backgroundColor: '#1D9E75',
    padding: 16, borderRadius: 12, alignItems: 'center',
  },
  permBtnText: { color: '#fff', fontWeight: '700' },
})