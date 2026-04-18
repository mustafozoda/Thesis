import React, { useState } from 'react'
import {
  View, Text, StyleSheet, TouchableOpacity,
  TextInput, Switch, ScrollView, StatusBar, Alert,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useTheme } from '../context/ThemeContext'
import { useServer } from '../context/ServerContext'

export default function SettingsScreen({ navigation, route }) {
  const { theme: t, isDark, toggle } = useTheme()
  const s = makeStyles(t)

  const { server, saveServer } = useServer()
  const [url, setUrl] = useState(server)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null) // null | 'ok' | 'error'

  const testConnection = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 5000)
      const res = await fetch(`${url}/health`, { signal: controller.signal })
      clearTimeout(timer)
      if (res.ok) {
        const data = await res.json()
        setTestResult({ ok: true, models: data.models_count, device: data.device })
      } else {
        setTestResult({ ok: false })
      }
    } catch (e) {
      setTestResult({ ok: false, error: e.message })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = () => {
    if (!url.startsWith('http')) {
      Alert.alert('Invalid URL', 'URL must start with http:// or https://')
      return
    }
    saveServer(url.replace(/\/$/, ''))
    navigation.goBack()
  }

  return (
    <SafeAreaView style={[s.root, { backgroundColor: t.bg }]}>
      <StatusBar barStyle={t.statusBar} backgroundColor={t.bg} />

      {/* Header */}
      <View style={s.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={s.backBtn}>
          <Text style={[s.backArrow, { color: t.textPrimary }]}>←</Text>
        </TouchableOpacity>
        <Text style={[s.headerTitle, { color: t.textPrimary }]}>Settings</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView contentContainerStyle={s.scroll} showsVerticalScrollIndicator={false}>

        {/* Server section */}
        <Text style={[s.sectionLabel, { color: t.textMuted }]}>SERVER</Text>
        <View style={[s.card, { backgroundColor: t.card, borderColor: t.border }]}>
          <Text style={[s.fieldLabel, { color: t.textSec }]}>Server URL</Text>
          <TextInput
            style={[s.input, { color: t.textPrimary, backgroundColor: t.cardAlt, borderColor: t.borderMid }]}
            value={url}
            onChangeText={setUrl}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            placeholder="http://192.168.1.x:8000"
            placeholderTextColor={t.textTertiary}
          />
          <TouchableOpacity
            style={[s.testBtn, { backgroundColor: t.accentDim, borderColor: t.accentBorder }]}
            onPress={testConnection}
            disabled={testing}
          >
            <Text style={[s.testBtnText, { color: t.accent }]}>
              {testing ? 'Testing...' : 'Test Connection'}
            </Text>
          </TouchableOpacity>

          {testResult && (
            <View style={[
              s.testResult,
              {
                backgroundColor: testResult.ok ? '#1D9E7515' : '#FF507815',
                borderColor: testResult.ok ? '#1D9E7535' : '#FF507835',
              }
            ]}>
              <Text style={{ color: testResult.ok ? '#1D9E75' : '#FF5078', fontSize: 13, fontWeight: '600' }}>
                {testResult.ok
                  ? `✓ Connected — ${testResult.models} models on ${testResult.device}`
                  : `✗ Failed${testResult.error ? ': ' + testResult.error : ''}`}
              </Text>
            </View>
          )}
        </View>

        {/* Appearance */}
        <Text style={[s.sectionLabel, { color: t.textMuted }]}>APPEARANCE</Text>
        <View style={[s.card, { backgroundColor: t.card, borderColor: t.border }]}>
          <View style={s.row}>
            <View>
              <Text style={[s.rowTitle, { color: t.textPrimary }]}>Dark Mode</Text>
              <Text style={[s.rowSub, { color: t.textTertiary }]}>Switch between light and dark theme</Text>
            </View>
            <Switch
              value={isDark}
              onValueChange={toggle}
              trackColor={{ true: t.accent, false: t.switchTrackOff }}
              thumbColor="#fff"
            />
          </View>
        </View>

        {/* App info */}
        <Text style={[s.sectionLabel, { color: t.textMuted }]}>ABOUT</Text>
        <View style={[s.card, { backgroundColor: t.card, borderColor: t.border }]}>
          {[
            ['Version', '2.0.0'],
            ['Models', '6 (MobileNetV2 + EfficientNet-B0)'],
            ['Classes', '7 (Background + 6 Tomato states)'],
            ['Best mIoU', '0.76 (Step 2 EfficientNet-B0)'],
          ].map(([k, v]) => (
            <View key={k} style={[s.infoRow, { borderBottomColor: t.border }]}>
              <Text style={[s.infoKey, { color: t.textSec }]}>{k}</Text>
              <Text style={[s.infoVal, { color: t.textPrimary }]}>{v}</Text>
            </View>
          ))}
        </View>

      </ScrollView>

      {/* Save button */}
      <View style={[s.footer, { backgroundColor: t.bgSecondary, borderTopColor: t.border }]}>
        <TouchableOpacity style={[s.saveBtn, { backgroundColor: t.accent }]} onPress={handleSave}>
          <Text style={s.saveBtnText}>Save Settings</Text>
        </TouchableOpacity>
      </View>
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
  backBtn: { padding: 4, width: 40 },
  backArrow: { fontSize: 22 },
  headerTitle: { fontSize: 17, fontWeight: '700', letterSpacing: -0.3 },
  scroll: { padding: 16, paddingBottom: 32 },
  sectionLabel: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1.3,
    marginBottom: 8,
    marginTop: 20,
    marginLeft: 4,
  },
  card: {
    borderRadius: 14,
    borderWidth: 1,
    padding: 16,
    gap: 12,
  },
  fieldLabel: { fontSize: 12, fontWeight: '600', letterSpacing: 0.3 },
  input: {
    borderRadius: 10,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 14,
    fontFamily: 'monospace',
  },
  testBtn: {
    paddingVertical: 12,
    borderRadius: 10,
    borderWidth: 1,
    alignItems: 'center',
  },
  testBtnText: { fontSize: 14, fontWeight: '600' },
  testResult: {
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  rowTitle: { fontSize: 15, fontWeight: '500' },
  rowSub: { fontSize: 12, marginTop: 2 },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
  },
  infoKey: { fontSize: 13 },
  infoVal: { fontSize: 13, fontWeight: '600' },
  footer: {
    padding: 16,
    borderTopWidth: 1,
  },
  saveBtn: {
    paddingVertical: 15,
    borderRadius: 13,
    alignItems: 'center',
  },
  saveBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
})