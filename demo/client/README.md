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
   ├── context/
   │   └── ThemeContext.js
   ├── screens/
   │   ├── MenuScreen.js
   │   └── CameraScreen.js
   └── components/
      └── CoverageBars.js
```

---

## Features

- **Live mode** — continuous inference every 1.5 seconds over the camera feed
- **Photo mode** — capture a photo and analyse it on demand
- **Upload mode** — pick an image from the phone gallery and analyse it
- **Overlay toggle** — switch between original and segmented view
- **Dark / Light theme** — toggle between dark and light UI from the menu
- **Ripeness coverage bars** — shows fully ripened / half ripened / green as % of tomato pixels only
- **Model selector** — choose between 6 trained models (3 steps × 2 encoders)

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
  ✅ Step3 — EfficientNet-B0 (Synthetic)
INFO: Uvicorn running on http://0.0.0.0:8000
```

Find your laptop's IP address:

```bash
ipconfig        # Windows
ifconfig        # Mac / Linux
```

Look for the WiFi adapter or hotspot IP (e.g. `192.168.137.1`).

### 2. Client Setup

```bash
cd demo/client
npm install
```

Update the server IP in `screens/MenuScreen.js`:

```javascript
const SERVER = "http://YOUR_LAPTOP_IP:8000";
```

Start the Expo development server:

```bash
npx expo start
```

---

## Phone Setup (Expo Go)

### Step 1 — Install Expo Go

On your Android phone open the **Google Play Store**, search for **Expo Go** and install it.

### Step 2 — Set up laptop hotspot

Your phone and laptop must be on the same local network. The easiest way is your laptop's hotspot:

**On Windows:**

1. Open Settings → Network & Internet → Mobile Hotspot
2. Turn it on
3. Connect your phone to the hotspot using the shown network name and password
4. Your laptop IP on the hotspot will be `192.168.137.1`

> University or corporate WiFi networks isolate devices from each other — always use a personal hotspot instead.

### Step 3 — Open the app

1. Make sure `npx expo start` is running on your laptop
2. Open **Expo Go** on your phone
3. Tap **Scan QR code**
4. Scan the QR code shown in the terminal or browser
5. The app will bundle and load on your phone

> If the QR scan doesn't work, in Expo Go tap **Enter URL manually** and type the address shown in the terminal (e.g. `exp://192.168.137.1:8081`)

---

## Network Setup Summary

| Component            | Value                             |
| -------------------- | --------------------------------- |
| Laptop hotspot IP    | `192.168.137.1` (Windows default) |
| Server port          | `8000`                            |
| Expo dev server port | `8081`                            |
| SERVER value in app  | `http://192.168.137.1:8000`       |

---

## How It Works

```
Phone camera
    ↓
Takes a photo (live: every 1.5s, photo: on demand)
    ↓
Sends JPEG to FastAPI server over WiFi
    ↓
Server crops center square → resizes to 512×512 → runs U-Net model
    ↓
Returns segmentation overlay (base64 JPEG) + class coverage %
    ↓
App displays overlay on camera feed + ripeness coverage bars
```

### Coverage Display

Coverage is shown as **percentage of tomato pixels only** (background excluded):

**RIPENESS**

- Fully ripened = `b_fully_ripened` + `l_fully_ripened`
- Half ripened = `b_half_ripened` + `l_half_ripened`
- Green = `b_green` + `l_green`

---

## API Endpoints

### `GET /models`

Returns list of loaded models.

```json
{
  "models": [
    "Step1 — MobileNetV2 (Natural)",
    "Step2 — EfficientNet-B0 (Removed)",
    "..."
  ]
}
```

### `POST /segment?model_name=<n>`

Accepts a JPEG image, returns overlay and coverage.

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
  }
}
```

---

## Performance

| Metric                  | Value        |
| ----------------------- | ------------ |
| Inference device        | CPU (laptop) |
| Average latency         | ~3–8 seconds |
| Live inference interval | 1.5 seconds  |
| Model input size        | 512 × 512    |

> Latency can be reduced by setting `IMG_SIZE = 256` in `server.py` at the cost of some segmentation detail.

---

## Troubleshooting

| Problem                       | Fix                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------- |
| "Cannot reach server" on menu | Hotspot is off or server not running — check both                               |
| App won't load on phone       | Make sure phone is on laptop hotspot, not mobile data                           |
| QR code scan fails            | In Expo Go tap "Enter URL manually" and type the `exp://` address from terminal |
| Models not listed             | Check `.pth` files are inside `server/models/` folder                           |
| Result overlay misaligned     | Server crops center square automatically — this is expected                     |
| Segmentation looks poor       | Use Step 2 EfficientNet-B0 — best model with mIoU 0.7646                        |

---

## Dataset

**Laboro Tomato Dataset** — semantic segmentation dataset with polygon annotations for 7 tomato classes across 643 farm images.

- 643 total images
- 70% train / 15% validation / 15% test split
- Stratified split by dominant class

---

## Thesis Context

This app demonstrates the key finding of the thesis: **background bias significantly affects segmentation performance**.

- Step 1 — natural background: mIoU 0.63
- Step 2 — background removed during training: mIoU 0.76 → **+13% improvement**
- Step 3 — synthetic background: mIoU 0.75

Removing background from training images forces the model to learn tomato features rather than farm background cues, leading to substantially better segmentation quality.
