from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import io
import base64
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


print("Loading models...")
models = {}
for name, (enc, path) in MODEL_CONFIGS.items():
    try:
        models[name] = load_model(enc, path)
        print(f"  ✅ {name}")
    except FileNotFoundError:
        print(f"  ⚠️  Not found: {path}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/models")
def get_models():
    return {"models": list(models.keys())}


@app.post("/segment")
async def segment(model_name: str, file: UploadFile = File(...)):
    if model_name not in models:
        raise HTTPException(404, f"Model not found: {model_name}")

    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    img_np = np.array(img)
    # img_np = cv2.flip(img_np, 0)

    # Crop center square from the full photo
    img_h, img_w = img_np.shape[:2]
    if img_h != img_w:
        if img_h > img_w:
            # portrait — crop top and bottom
            start = (img_h - img_w) // 2
            img_np = img_np[start:start + img_w, :, :]
        else:
            # landscape — crop left and right
            start = (img_w - img_h) // 2
            img_np = img_np[:, start:start + img_h, :]

    t = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE),
                   A.Normalize(mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225)),
                   ToTensorV2()])
    tensor = t(image=img_np)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = models[model_name](tensor).argmax(
            dim=1).squeeze(0).cpu().numpy()

    colored = colorize(mask)
    h, w = img_np.shape[:2]
    img_r = cv2.resize(img_np, (w, h))
    mask_colored_r = cv2.resize(colored, (w, h))
    blended = cv2.addWeighted(cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR), 0.45,
                              cv2.cvtColor(mask_colored_r, cv2.COLOR_RGB2BGR), 0.55, 0)

    _, buf = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()

    total = mask.size
    coverage = {CLASS_NAMES[c]: round(float((mask == c).sum()/total*100), 1)
                for c in range(len(CLASS_NAMES))}

    return JSONResponse({"overlay_b64": b64, "coverage": coverage})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
