# TomatoSeg v2.0 — Tomato Quality Segmentation App

A thesis demo application for real-time tomato quality segmentation using deep learning. The system consists of a Python FastAPI server running U-Net segmentation models and a React Native mobile app for live camera analysis, photo capture, and gallery upload.

---

## Project Overview

This app was built as part of a thesis on **background bias analysis in semantic segmentation** for tomato quality assessment. It classifies tomatoes into 7 categories based on size and ripeness using trained U-Net models with MobileNetV2 and EfficientNet-B0 encoders.

### Classes

| Class             | Description                    |
| ----------------- | ------------------------------ |
| `background`      | Non-tomato pixels              |
| `b_fully_ripened` | Big tomato — fully ripened     |
| `b_half_ripened`  | Big tomato — half ripened      |
| `b_green`         | Big tomato — green (unripe)    |
| `l_fully_ripened` | Little tomato — fully ripened  |
| `l_half_ripened`  | Little tomato — half ripened   |
| `l_green`         | Little tomato — green (unripe) |

### Models

| Model  | Encoder         | Training             | mIoU       |
| ------ | --------------- | -------------------- | ---------- |
| Step 1 | MobileNetV2     | Natural background   | 0.6352     |
| Step 1 | EfficientNet-B0 | Natural background   | 0.6323     |
| Step 2 | MobileNetV2     | Background removed   | 0.7255     |
| Step 2 | EfficientNet-B0 | Background removed   | **0.7646** |
| Step 3 | MobileNetV2     | Synthetic background | 0.6995     |
| Step 3 | EfficientNet-B0 | Synthetic background | 0.7512     |

---

## Project Structure

```
demo/
├── server/
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│       ├── step1_mobilenetv2_natural_best.pth
│       ├── step1_efficientnetb0_natural_best.pth
│       ├── step2_mobilenetv2_removed_best.pth
│       ├── step2_efficientnetb0_removed_best.pth
│       ├── step3_mobilenetv2_synthetic_best.pth
│       └── step3_efficientnetb0_synthetic_best.pth
│
└── client/
    ├── App.js
    ├── index.js
    ├── package.json
    ├── app.json
    ├── context/
    │   ├── ThemeContext.js
    │   └── ScanHistoryContext.js
    ├── components/
    │   ├── CoverageBars.js
    │   ├── ServerContext.js
    │   └── ExportCard.js
    └── screens/
        ├── OnboardingScreen.js
        ├── MenuScreen.js
        ├── CameraScreen.js
        ├── HistoryScreen.js
        ├── SplashScreen.js
        └── SettingsScreen.js
```

---

## Features

### Analysis Modes

- **Live mode** — continuous inference every 1.5 seconds over the camera feed
- **Photo mode** — capture a photo and analyse it on demand
- **Upload mode** — pick an image from the phone gallery and analyse it

### UI & UX

- **Splash screen** — animated logo on app launch
- **Onboarding** — 4-slide intro shown once on first launch explaining the app
- **Dark / Light theme** — toggle from the menu header, persisted across sessions
- **Animated scan line** — sweeps across viewport in live mode
- **Smooth overlay fade** — segmentation overlay fades in when result arrives
- **Class legend** — tap ⬡ to show all 7 class colors on the viewport
- **Inference time badge** — shows server inference time on still image results

### Results & Coverage

- **Ripeness coverage bars** — animated bars showing fully ripened / half ripened / green as % of tomato pixels
- **Confidence scores** — per-class model confidence shown alongside coverage
- **Dominant ripeness badge** — highlights the most prevalent ripeness group

### History & Export

- **Scan history** — every photo/upload scan is automatically saved with thumbnail, coverage, model, and timestamp
- **History screen** — browse past scans, tap for detail view, long press to delete, clear all option
- **Export card** — generates a summary card image (overlay + stats) and shares via Android share sheet

### Settings

- **Configurable server URL** — change and test server connection from the Settings screen
- **Persistent server URL** — saved across app restarts via AsyncStorage
- **Connection test** — verify server is reachable before scanning

### Model Selection

- Choose between 6 trained models (3 steps × 2 encoders)
- Info bar shows mIoU for the selected model
- Step badge (S1/S2/S3) color coded throughout the app

---

## Setup

### 1. Server Setup

Place all `.pth` model files inside `server/models/`, then install dependencies:

```bash
cd demo/server
pip install -r requirements.txt
```

Start the server:

```bash
python main.py
```

You should see:

```
Loading models...
  ✅ Step1 — MobileNetV2 (Natural)
  ✅ Step1 — EfficientNet-B0 (Natural)
  ✅ Step2 — MobileNetV2 (Removed)
  ✅ Step2 — EfficientNet-B0 (Removed)
  ✅ Step3 — MobileNetV2 (Synthetic)
  ✅ Step3 — EfficientNet-B0 (Synthetic)
INFO: Uvicorn running on http://0.0.0.0:8000
```

Find your laptop's IP address:

```bash
ipconfig        # Windows
ifconfig        # Mac / Linux
```

Look for the WiFi adapter or hotspot IP (e.g. `192.168.137.1`).

---

### 2. Client Setup

```bash
cd demo/client
npm install
```

#### Option A — Development build (recommended)

Connect your Android phone via USB with USB debugging enabled, then:

```bash
npx expo run:android --device
```

The app will build and install directly on your phone. After installation USB can be disconnected — the app stays installed permanently.

#### Option B — Release APK

For a fast-opening optimized build:

```bash
cd android
.\gradlew.bat assembleRelease
```

APK will be at:

```
android/app/build/outputs/apk/release/app-release.apk
```

Send to your phone via WhatsApp, Google Drive, or email and install it.

---

### 3. Network Setup

Your phone and laptop must be on the same local network. The easiest way is your laptop's mobile hotspot.

**On Windows:**

1. Settings → Network & Internet → Mobile Hotspot → Turn on
2. Connect your phone to the hotspot
3. Your laptop's hotspot IP is `192.168.137.1` by default

**In the app:**

- Open the app → tap ⚙ Settings
- Enter your server URL (e.g. `http://192.168.137.1:8000`)
- Tap **Test Connection** to verify
- Tap **Save Settings**

The URL is remembered across app restarts.

> University or corporate WiFi networks isolate devices — always use a personal hotspot.

---

## Network Summary

| Component          | Value                             |
| ------------------ | --------------------------------- |
| Laptop hotspot IP  | `192.168.137.1` (Windows default) |
| Server port        | `8000`                            |
| Default server URL | `http://192.168.137.1:8000`       |

---

## API Endpoints

### `GET /health`

Returns server status, loaded models, and device info.

```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": ["Step1 — MobileNetV2 (Natural)", "..."],
  "models_count": 6,
  "timestamp": 1234567890.0
}
```

### `GET /models`

Returns list of available loaded models.

```json
{
  "models": [
    "Step1 — MobileNetV2 (Natural)",
    "Step2 — EfficientNet-B0 (Removed)",
    "..."]
  ]
}
```

### `POST /segment?model_name=<name>`

Accepts a JPEG image, returns overlay, coverage, and confidence scores.

**Request:** `multipart/form-data` with `file` field (JPEG)

**Response:**

```json
{
  "overlay_b64": "<base64 JPEG>",
  "coverage": {
    "background": 45.2,
    "b_fully_ripened": 12.3,
    "b_half_ripened": 8.1,
    "b_green": 5.4,
    "l_fully_ripened": 9.7,
    "l_half_ripened": 6.2,
    "l_green": 13.1
  },
  "confidence": {
    "background": 94.2,
    "b_fully_ripened": 87.5,
    "b_half_ripened": 81.3,
    "b_green": 90.1,
    "l_fully_ripened": 85.6,
    "l_half_ripened": 79.8,
    "l_green": 88.4
  },
  "inference_ms": 42.3
}
```

### `POST /segment/batch?model_name=<name>`

Accepts up to 10 images at once, returns results for each.

**Request:** `multipart/form-data` with multiple `files` fields

**Response:**

```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "overlay_b64": "...",
      "coverage": { "..." },
      "confidence": { "..." }
    }
  ],
  "count": 1
}
```

---

## How It Works

```
Phone camera / gallery
        ↓
Takes photo (live: every 1.5s, photo/upload: on demand)
        ↓
Sends JPEG to FastAPI server over WiFi
        ↓
Server: crop center square → resize 512×512 → U-Net inference
        ↓
Returns overlay (base64 JPEG) + coverage % + confidence % + inference_ms
        ↓
App: displays overlay on viewport + ripeness bars + confidence scores
        ↓
Scan saved to history (photo/upload mode only)
```

### Coverage Display

Coverage is shown as **percentage of tomato pixels only** (background excluded):

- **Fully Ripened** = `b_fully_ripened` + `l_fully_ripened`
- **Half Ripened** = `b_half_ripened` + `l_half_ripened`
- **Green** = `b_green` + `l_green`

---

## Performance

| Metric                  | Value          |
| ----------------------- | -------------- |
| Inference device        | CPU (laptop)   |
| Average latency         | ~1.5–5 seconds |
| Live inference interval | 1.5 seconds    |
| Model input size        | 512 × 512      |
| Max batch size          | 10 images      |

> Latency can be reduced by setting `IMG_SIZE = 256` in `main.py` at the cost of some segmentation detail.

---

## Troubleshooting

| Problem                       | Fix                                                                |
| ----------------------------- | ------------------------------------------------------------------ |
| "Cannot reach server" on menu | Hotspot is off or server not running — check both                  |
| Test Connection fails         | Make sure phone is on laptop hotspot, check server URL in Settings |
| App opens slowly              | Use release APK build instead of debug build                       |
| Emulator crashes on build     | Connect physical phone via USB and use `--device` flag             |
| Models not listed             | Check `.pth` files are inside `server/models/` folder              |
| Export card fails             | Use share sheet option — full gallery save requires release build  |
| Segmentation looks poor       | Use Step 2 EfficientNet-B0 — best model with mIoU 0.7646           |
| Result overlay misaligned     | Server crops center square automatically — expected behavior       |

---

## Dataset

**Laboro Tomato Dataset** — semantic segmentation dataset with polygon annotations for 7 tomato classes across 643 farm images.

- 643 total images
- 70% train / 15% validation / 15% test split
- Stratified split by dominant class

---

## Thesis Context

This app demonstrates the key finding of the thesis: **background bias significantly affects segmentation performance**.

| Step   | Training Condition   | Best mIoU  | Change   |
| ------ | -------------------- | ---------- | -------- |
| Step 1 | Natural background   | 0.6352     | Baseline |
| Step 2 | Background removed   | **0.7646** | **+13%** |
| Step 3 | Synthetic background | 0.7512     | +11.6%   |

Removing background from training images forces the model to learn tomato features rather than farm background cues, leading to substantially better segmentation quality. Step 3 demonstrates that synthetic backgrounds provide robustness comparable to background removal, suggesting a practical alternative when background removal is not feasible.
