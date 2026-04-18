from fastapi import FastAPI, File, UploadFile, HTTPException, Query  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore
import io
import base64
import time
from PIL import Image  # type: ignore
import segmentation_models_pytorch as smp  # type: ignore
import albumentations as A  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore
from typing import List, Optional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512

CLASS_NAMES = ['background', 'b_fully_ripened', 'b_half_ripened', 'b_green',
               'l_fully_ripened', 'l_half_ripened', 'l_green']
CLASS_COLORS = [(0, 0, 0), (255, 80, 120), (80, 120, 255), (80, 255, 180),
                (255, 140, 80), (120, 80, 255), (80, 200, 80)]

MODEL_CONFIGS = {
    'Step1 — MobileNetV2 (Natural)':     ('mobilenet_v2',    'models/step1_mobilenetv2_natural_best.pth'),
    'Step1 — EfficientNet-B0 (Natural)': ('efficientnet-b0', 'models/step1_efficientnetb0_natural_best.pth'),
    'Step2 — MobileNetV2 (Removed)':     ('mobilenet_v2',    'models/step2_mobilenetv2_removed_best.pth'),
    'Step2 — EfficientNet-B0 (Removed)': ('efficientnet-b0', 'models/step2_efficientnetb0_removed_best.pth'),
    'Step3 — MobileNetV2 (Synthetic)':   ('mobilenet_v2',    'models/step3_mobilenetv2_synthetic_best.pth'),
    'Step3 — EfficientNet-B0 (Synth)':   ('efficientnet-b0', 'models/step3_efficientnetb0_synthetic_best.pth'),
}


def load_model(encoder, path):
    m = smp.Unet(encoder_name=encoder, encoder_weights=None,
                 in_channels=3, classes=7, activation=None)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()


def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        out[mask == c] = col
    return out


def crop_center_square(img_np):
    img_h, img_w = img_np.shape[:2]
    if img_h != img_w:
        if img_h > img_w:
            start = (img_h - img_w) // 2
            img_np = img_np[start:start + img_w, :, :]
        else:
            start = (img_w - img_h) // 2
            img_np = img_np[:, start:start + img_h, :]
    return img_np


def preprocess_image(img_np):
    t = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return t(image=img_np)['image'].unsqueeze(0).to(DEVICE)


def run_inference(model, img_np):
    tensor = preprocess_image(img_np)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        prob_np = probs.squeeze(0).cpu().numpy()
    return mask, prob_np


def build_response(img_np, mask, prob_np):
    colored = colorize(mask)
    h, w = img_np.shape[:2]
    img_r = cv2.resize(img_np, (w, h))
    mask_colored_r = cv2.resize(
        colored, (w, h), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(
        cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR), 0.45,
        cv2.cvtColor(mask_colored_r, cv2.COLOR_RGB2BGR), 0.55, 0
    )
    _, buf = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()

    total = mask.size
    coverage = {CLASS_NAMES[c]: round(float((mask == c).sum() / total * 100), 1)
                for c in range(len(CLASS_NAMES))}

    # Per-class mean confidence (only over pixels predicted as that class)
    confidence = {}
    for c in range(len(CLASS_NAMES)):
        pixel_mask = (mask == c)
        if pixel_mask.sum() > 0:
            mean_conf = float(prob_np[c][pixel_mask].mean())
        else:
            mean_conf = 0.0
        confidence[CLASS_NAMES[c]] = round(mean_conf * 100, 1)

    return {"overlay_b64": b64, "coverage": coverage, "confidence": confidence}


print("Loading models...")
models = {}
for name, (enc, path) in MODEL_CONFIGS.items():
    try:
        models[name] = load_model(enc, path)
        print(f"  ✅ {name}")
    except FileNotFoundError:
        print(f"  ⚠️  Not found: {path}")

app = FastAPI(title="Tomato Segmentation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": list(models.keys()),
        "models_count": len(models),
        "timestamp": time.time()
    }


@app.get("/models")
def get_models():
    return {"models": list(models.keys())}


@app.post("/segment")
async def segment(model_name: str, file: UploadFile = File(...)):
    if model_name not in models:
        raise HTTPException(404, f"Model not found: {model_name}")

    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    img_np = np.array(img)
    img_np = crop_center_square(img_np)

    t0 = time.time()
    mask, prob_np = run_inference(models[model_name], img_np)
    inference_ms = round((time.time() - t0) * 1000, 1)

    result = build_response(img_np, mask, prob_np)
    result["inference_ms"] = inference_ms

    return JSONResponse(result)


@app.post("/segment/batch")
async def segment_batch(
    model_name: str,
    files: List[UploadFile] = File(...)
):
    if model_name not in models:
        raise HTTPException(404, f"Model not found: {model_name}")
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 images per batch")

    results = []
    for file in files:
        img = Image.open(io.BytesIO(await file.read())).convert('RGB')
        img_np = np.array(img)
        img_np = crop_center_square(img_np)
        mask, prob_np = run_inference(models[model_name], img_np)
        result = build_response(img_np, mask, prob_np)
        result["filename"] = file.filename
        results.append(result)

    return JSONResponse({"results": results, "count": len(results)})


if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
