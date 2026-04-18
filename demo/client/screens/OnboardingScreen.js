import React, { useRef, useState } from 'react'
import {
  View, Text, StyleSheet, TouchableOpacity,
  Dimensions, FlatList, StatusBar,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTheme } from '../context/ThemeContext'

const { width: W } = Dimensions.get('window')

const SLIDES = [
  {
    id: '1',
    icon: '🍅',
    title: 'Tomato Segmentation',
    subtitle: 'AI-powered quality analysis',
    body: 'Identify and classify tomatoes by ripeness in real-time using semantic segmentation. Supports live camera, photo capture, and gallery upload.',
    accent: '#1D9E75',
  },
  {
    id: '2',
    icon: '📊',
    title: 'Three Training Steps',
    subtitle: 'Progressive background complexity',
    body: 'Step 1 uses natural farm backgrounds as a baseline.\nStep 2 trains with background removed for focused learning.\nStep 3 uses synthetic backgrounds for robustness.',
    accent: '#378ADD',
  },
  {
    id: '3',
    icon: '🎯',
    title: '7 Classes Detected',
    subtitle: 'Precise ripeness + type classification',
    body: 'Classifies tomatoes into: background, big & little tomatoes — each in fully ripened, half ripened, and green states. Best model achieves mIoU 0.76.',
    accent: '#EF9F27',
  },
  {
    id: '4',
    icon: '⚡',
    title: 'Real-time Inference',
    subtitle: 'Multiple analysis modes',
    body: 'Live mode analyzes every 1.5s. Photo mode captures a still for detailed analysis. Upload mode lets you analyze any image from your gallery.',
    accent: '#FF5078',
  },
]

export default function OnboardingScreen({ onDone }) {
  const { theme: t } = useTheme()
  const s = makeStyles(t)
  const [currentIndex, setCurrentIndex] = useState(0)
  const flatRef = useRef(null)

  const scrollToIndex = (idx) => {
    flatRef.current?.scrollToIndex({ index: idx, animated: true })
    setCurrentIndex(idx)
  }

  const onViewableItemsChanged = useRef(({ viewableItems }) => {
    if (viewableItems[0]) {
      setCurrentIndex(viewableItems[0].index)
    }
  }).current

  const isLast = currentIndex === SLIDES.length - 1

  return (
    <SafeAreaView style={[s.root, { backgroundColor: t.bg }]}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* Skip */}
      <TouchableOpacity style={s.skipBtn} onPress={onDone}>
        <Text style={[s.skipText, { color: t.textTertiary }]}>Skip</Text>
      </TouchableOpacity>

      <FlatList
        ref={flatRef}
        data={SLIDES}
        keyExtractor={i => i.id}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onViewableItemsChanged={onViewableItemsChanged}
        viewabilityConfig={{ viewAreaCoveragePercentThreshold: 50 }}
        renderItem={({ item }) => (
          <View style={s.slide}>
            {/* Icon circle */}
            <View style={[s.iconCircle, { backgroundColor: item.accent + '18', borderColor: item.accent + '35' }]}>
              <Text style={s.icon}>{item.icon}</Text>
            </View>

            {/* Accent line */}
            <View style={[s.accentLine, { backgroundColor: item.accent }]} />

            <Text style={[s.slideTitle, { color: t.textPrimary }]}>{item.title}</Text>
            <Text style={[s.slideSubtitle, { color: item.accent }]}>{item.subtitle}</Text>
            <Text style={[s.slideBody, { color: t.textSec }]}>{item.body}</Text>
          </View>
        )}
      />

      {/* Dots */}
      <View style={s.dotsRow}>
        {SLIDES.map((_, i) => (
          <TouchableOpacity key={i} onPress={() => scrollToIndex(i)}>
            <View style={[
              s.dot,
              {
                backgroundColor: i === currentIndex ? SLIDES[i].accent : t.borderMid,
                width: i === currentIndex ? 24 : 8,
              }
            ]} />
          </TouchableOpacity>
        ))}
      </View>

      {/* CTA */}
      <TouchableOpacity
        style={[s.cta, { backgroundColor: SLIDES[currentIndex].accent }]}
        onPress={() => isLast ? onDone() : scrollToIndex(currentIndex + 1)}
        activeOpacity={0.85}
      >
        <Text style={s.ctaText}>{isLast ? 'Get Started' : 'Next'}</Text>
      </TouchableOpacity>
    </SafeAreaView>
  )
}

const makeStyles = (t) => StyleSheet.create({
  root: { flex: 1 },
  skipBtn: {
    position: 'absolute',
    top: 56,
    right: 24,
    zIndex: 10,
    padding: 8,
  },
  skipText: {
    fontSize: 14,
    fontWeight: '500',
  },
  slide: {
    width: W,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 36,
    paddingTop: 60,
    gap: 12,
  },
  iconCircle: {
    width: 100,
    height: 100,
    borderRadius: 30,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  icon: { fontSize: 44 },
  accentLine: {
    width: 36,
    height: 3,
    borderRadius: 2,
    marginBottom: 4,
  },
  slideTitle: {
    fontSize: 26,
    fontWeight: '700',
    letterSpacing: -0.5,
    textAlign: 'center',
  },
  slideSubtitle: {
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 0.3,
    textAlign: 'center',
  },
  slideBody: {
    fontSize: 14,
    lineHeight: 22,
    textAlign: 'center',
    marginTop: 8,
  },
  dotsRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 6,
    paddingBottom: 24,
  },
  dot: {
    height: 8,
    borderRadius: 4,
  },
  cta: {
    marginHorizontal: 24,
    marginBottom: 32,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
  },
  ctaText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
})