# TomatoSeg — Tomato Quality Segmentation App

A thesis demo application for real-time tomato quality segmentation using deep learning. The system consists of a Python FastAPI server running U-Net segmentation models and a React Native mobile app for live camera analysis.

---

## Project Overview

This app was built as part of a thesis on **background bias analysis in semantic segmentation** for tomato quality assessment. It classifies tomatoes into 7 categories based on size and ripeness using trained U-Net models with MobileNetV2 and EfficientNet-B0 encoders.

### Classes

| Class             | Description                    |
| ----------------- | ------------------------------ |
| `background`      | Non-tomato pixels              |
| `b_fully_ripened` | Big tomato — fully ripe        |
| `b_half_ripened`  | Big tomato — half ripe         |
| `b_green`         | Big tomato — green (unripe)    |
| `l_fully_ripened` | Little tomato — fully ripe     |
| `l_half_ripened`  | Little tomato — half ripe      |
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
│   ├── server.py
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
    ├── screens/
    │   ├── MenuScreen.js
    │   └── CameraScreen.js
    └── components/
        └── CoverageBars.js
```

---

## Requirements

### Server (Laptop)

- Python 3.10+
- CUDA-capable GPU (optional, CPU works but slower)
- Windows / Mac / Linux

### Client (Android Phone)

- Android 8.0+
- Expo Go app installed
- Same WiFi network as laptop (or connected via laptop hotspot)

---

## Setup & Installation

### 1. Server Setup

```bash
cd demo/server

pip install fastapi uvicorn torch segmentation-models-pytorch albumentations pillow python-multipart opencv-python
```

Place all `.pth` model files inside `server/models/`.

Start the server:

```bash
python server.py
```

You should see:

```
Loading models...
  ✅ Step1 — MobileNetV2 (Natural)
  ✅ Step1 — EfficientNet-B0 (Natural)
  ✅ Step2 — MobileNetV2 (Removed)
  ✅ Step2 — EfficientNet-B0 (Removed)
  ✅ Step3 — MobileNetV2 (Synthetic)
  ✅ Step3 — EfficientNet-B0 (Synth)
INFO: Uvicorn running on http://0.0.0.0:8000
```

Find your laptop IP address:

```bash
ipconfig        # Windows
ifconfig        # Mac / Linux
```

Look for the WiFi adapter IP (e.g. `192.168.137.1`).

### 2. Client Setup

```bash
cd demo/client/TomatoSeg

npm install
npx expo install expo-camera
npm install @react-navigation/native @react-navigation/stack react-native-screens react-native-safe-area-context
```

Update the server IP in `screens/MenuScreen.js`:

```javascript
const SERVER = "http://YOUR_LAPTOP_IP:8000";
```

Start the app:

```bash
npx expo start
```

Scan the QR code with **Expo Go** on your Android phone.

---

## Network Setup

This app requires the phone and laptop to be on the same local network.

### Recommended: Laptop Hotspot

1. Enable Mobile Hotspot on your laptop (Settings → Network → Mobile Hotspot)
2. Connect your phone to the laptop's hotspot
3. The laptop IP is typically `192.168.137.1` on Windows
4. Update `SERVER` in `MenuScreen.js` with this IP

> **Note:** University / corporate WiFi networks isolate devices from each other. Use a personal hotspot instead.

---

## How It Works

```
Phone camera
    ↓
Takes a photo (live mode: every 1.5s, photo mode: on demand)
    ↓
Sends JPEG to FastAPI server over WiFi
    ↓
Server crops center square → resizes to 512×512 → runs U-Net model
    ↓
Returns: segmentation overlay image (base64 JPEG) + class coverage %
    ↓
App displays overlay on camera feed + coverage bars
```

### Live Mode

- Camera runs continuously as a live preview
- A photo is taken silently every 1.5 seconds
- Result overlay is shown on top of the live feed
- Coverage bars update with each result

### Photo Mode

- Press the shutter button to capture a photo
- "Analysing..." indicator shown while processing
- Full result overlay displayed on the captured photo
- Toggle overlay on/off to compare original vs segmented

---

## Coverage Display

The app shows two sections of coverage statistics, calculated as **percentage of tomato pixels only** (background excluded):

**RIPENESS**

- Fully ripened = big fully ripe + little fully ripe
- Half ripened = big half ripe + little half ripe
- Green = big green + little green

**SIZE**

- Big tomatoes = all `b_` classes combined
- Little tomatoes = all `l_` classes combined

---

## API Endpoints

### `GET /models`

Returns list of available loaded models.

```json
{
  "models": [
    "Step1 — MobileNetV2 (Natural)",
    "Step2 — EfficientNet-B0 (Removed)",
    ...
  ]
}
```

### `POST /segment?model_name=<name>`

Accepts a JPEG image, returns segmentation overlay and coverage.

**Request:** `multipart/form-data` with `file` field (JPEG image)

**Response:**

```json
{
  "overlay_b64": "<base64 encoded JPEG>",
  "coverage": {
    "background": 45.2,
    "b_fully_ripened": 12.3,
    "b_half_ripened": 8.1,
    "b_green": 5.4,
    "l_fully_ripened": 9.7,
    "l_half_ripened": 6.2,
    "l_green": 13.1
  }
}
```

---

## Performance

| Metric                  | Value                                      |
| ----------------------- | ------------------------------------------ |
| Inference device        | CPU (laptop)                               |
| Average latency         | ~3–8 seconds                               |
| Image size sent         | Full camera resolution (cropped to square) |
| Model input size        | 512 × 512                                  |
| Live inference interval | 1.5 seconds                                |

> Latency can be reduced by setting `IMG_SIZE = 256` in `server.py` at the cost of some segmentation detail.

---

## Troubleshooting

| Problem                                | Fix                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------ |
| "Cannot reach server"                  | Check hotspot is on, server is running, IP is correct in `MenuScreen.js` |
| Models show "Not found"                | Make sure `.pth` files are inside `server/models/` folder                |
| App shows "Something went wrong"       | Phone is using mobile data instead of WiFi — disable mobile data         |
| Result image doesn't match camera view | Server crops center square from full photo to match square viewport      |
| Segmentation looks wrong               | Try Step 2 EfficientNet-B0 — it has the best mIoU (0.7646)               |

---

## Dataset

**Laboro Tomato Dataset** — semantic segmentation dataset with polygon annotations for 7 tomato classes across 643 images captured in farm conditions.

- 643 total images
- 70% train / 15% validation / 15% test split
- Stratified by dominant class

---

## Thesis Context

This app demonstrates the key finding of the thesis: **background bias significantly affects segmentation model performance**.

- Step 1 (natural background): mIoU 0.63
- Step 2 (background removed during training): mIoU 0.76 — **+13% improvement**
- Step 3 (synthetic background): mIoU 0.75

Removing background information from training images forces the model to focus on tomato features rather than farm background cues, leading to substantially better segmentation quality.
