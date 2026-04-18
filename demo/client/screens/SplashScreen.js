import React, { useEffect, useRef } from 'react'
import { View, Text, StyleSheet, Animated, StatusBar, Dimensions } from 'react-native'
import { useTheme } from '../context/ThemeContext'

const { width: W } = Dimensions.get('window')

export default function SplashScreen({ onFinish }) {
  const { theme: t } = useTheme()

  const logoScale = useRef(new Animated.Value(0.6)).current
  const logoOpacity = useRef(new Animated.Value(0)).current
  const textOpacity = useRef(new Animated.Value(0)).current
  const barWidth = useRef(new Animated.Value(0)).current
  const containerOpacity = useRef(new Animated.Value(1)).current

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.timing(logoScale, { toValue: 1, duration: 500, useNativeDriver: true }),
        Animated.timing(logoOpacity, { toValue: 1, duration: 400, useNativeDriver: true }),
      ]),
      Animated.timing(textOpacity, { toValue: 1, duration: 300, delay: 100, useNativeDriver: true }),
      Animated.timing(barWidth, { toValue: W * 0.5, duration: 800, useNativeDriver: false }),
      Animated.delay(300),
      Animated.timing(containerOpacity, { toValue: 0, duration: 300, useNativeDriver: true }),
    ]).start(() => {
      if (onFinish) onFinish()
    })
  }, [barWidth, containerOpacity, logoOpacity, logoScale, onFinish, textOpacity])

  return (
    <Animated.View style={[styles.root, { backgroundColor: t.bg, opacity: containerOpacity }]}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* Logo mark */}
      <Animated.View style={[styles.logoWrap, { opacity: logoOpacity, transform: [{ scale: logoScale }] }]}>
        <View style={[styles.logoOuter, { borderColor: t.accent + '40' }]}>
          <View style={[styles.logoInner, { backgroundColor: t.accent + '20', borderColor: t.accent }]}>
            <View style={[styles.logoDot, { backgroundColor: t.accent }]} />
          </View>
        </View>
      </Animated.View>

      {/* Title */}
      <Animated.View style={{ opacity: textOpacity, alignItems: 'center' }}>
        <Text style={[styles.title, { color: t.textPrimary }]}>Tomato</Text>
        <Text style={[styles.subtitle, { color: t.textSec }]}>Quality Segmentation</Text>
      </Animated.View>

      {/* Loading bar */}
      <View style={[styles.barTrack, { backgroundColor: t.borderLight }]}>
        <Animated.View style={[styles.barFill, { width: barWidth, backgroundColor: t.accent }]} />
      </View>

      <Text style={[styles.version, { color: t.textMuted }]}>v2.0</Text>
    </Animated.View>
  )
}

const styles = StyleSheet.create({
  root: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 20,
    zIndex: 999,
  },
  logoWrap: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  logoOuter: {
    width: 96,
    height: 96,
    borderRadius: 28,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoInner: {
    width: 72,
    height: 72,
    borderRadius: 20,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
    letterSpacing: -1,
  },
  subtitle: {
    fontSize: 13,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginTop: 4,
  },
  barTrack: {
    width: W * 0.5,
    height: 2,
    borderRadius: 1,
    marginTop: 24,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 1,
  },
  version: {
    position: 'absolute',
    bottom: 48,
    fontSize: 12,
    letterSpacing: 0.5,
  },
})