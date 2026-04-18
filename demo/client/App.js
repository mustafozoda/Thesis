import React, { useState, useEffect } from 'react'
import { NavigationContainer } from '@react-navigation/native'
import { createStackNavigator } from '@react-navigation/stack'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import AsyncStorage from '@react-native-async-storage/async-storage'
import { ThemeProvider } from './context/ThemeContext'
import { ServerProvider } from './context/ServerContext'
import { ScanHistoryProvider } from './context/ScanHistoryContext'
import SplashScreen from './screens/SplashScreen'
import OnboardingScreen from './screens/OnboardingScreen'
import MenuScreen from './screens/MenuScreen'
import CameraScreen from './screens/CameraScreen'
import SettingsScreen from './screens/SettingsScreen'
import HistoryScreen from './screens/HistoryScreen'

const Stack = createStackNavigator()

function AppContent() {
  const [showSplash, setShowSplash] = useState(true)
  const [showOnboarding, setShowOnboarding] = useState(false)

  useEffect(() => {
    AsyncStorage.getItem('onboardingDone').then(val => {
      if (!val) setShowOnboarding(true)
    })
  }, [])

  const handleOnboardingDone = async () => {
    await AsyncStorage.setItem('onboardingDone', 'true')
    setShowOnboarding(false)
  }

  if (showSplash) {
    return <SplashScreen onFinish={() => setShowSplash(false)} />
  }

  if (showOnboarding) {
    return <OnboardingScreen onDone={handleOnboardingDone} />
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Menu" component={MenuScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="Settings" component={SettingsScreen} />
        <Stack.Screen name="History" component={HistoryScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  )
}

export default function App() {
  return (
    <SafeAreaProvider>
      <ThemeProvider>
        <ServerProvider>
          <ScanHistoryProvider>
            <AppContent />
          </ScanHistoryProvider>
        </ServerProvider>
      </ThemeProvider>
    </SafeAreaProvider>
  )
}