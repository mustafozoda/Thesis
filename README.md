# Evaluating Background Bias and Improving Model Reliability in Tomato Quality Classification Using Deep Learning and Explainable AI

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/sharifbek/tomato-quality-semantic-segmentation)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/mustafozoda/Thesis)

</div>

---

## Overview

This repository contains the full implementation of a Master's thesis
research project investigating how background information affects
tomato quality classification models using semantic segmentation.

The core research question is: **do models learn to classify tomatoes
based on their actual visual characteristics, or do they rely on
background cues from the farm environment?**

To answer this, a three-step experimental framework was designed:

- **Step 1 — Natural Background:** Models trained and tested on
  original unmodified farm images (baseline)
- **Step 2 — Background Removed:** All background pixels replaced
  with neutral gray (RGB: 128, 128, 128) using ground truth masks
- **Step 3 — Synthetic Backgrounds:** Tomato foregrounds composited
  onto procedurally generated synthetic backgrounds (8 texture types)

Two U-Net architectures were evaluated across all three conditions:

- U-Net + MobileNetV2 encoder (6.6M parameters)
- U-Net + EfficientNet-B0 encoder (6.3M parameters)

Cross-background robustness was also evaluated by testing models
on background conditions different from their training condition.

---

## Repository Structure

```text
THESIS/
├── tomato-seg-latest/          ← MAIN PROJECT (latest version)
│                                 Full pipeline, all 7 notebooks,
│                                 trained on Kaggle GPU (Tesla T4)
│
├── tomato-seg-latest-sep/      ← Local CPU version (Windows)
│                                 Same codebase, separated notebooks,
│                                 run locally for development/testing
│
├── docs/                       ← Thesis LaTeX source files
├── old-versions/               ← Early prototype (deprecated)
├── old-versions-v2/            ← Intermediate version (deprecated)
├── old-versions-v3/            ← Intermediate version (deprecated)
├── .gitignore
├── README.md
└── requirements.txt
```

> **The canonical version of this project is in `tomato-seg-latest/`.**
> This is the version that was executed on Kaggle and whose results
> are reported in the thesis. The `tomato-seg-latest-sep/` folder
> contains the same pipeline separated into individual notebooks
> for local CPU execution on Windows, used during development.

---

## Notebook Structure

The project is organized as a sequence of 7 notebooks:

| Notebook | Content                                                                             |
| -------- | ----------------------------------------------------------------------------------- |
| NB-01    | Setup, configuration, data splitting, and EDA                                       |
| NB-02    | Data pipeline — datasets, augmentations, DataLoaders                                |
| NB-03    | Model architecture and training utilities                                           |
| NB-04    | Step 1 — Natural background baseline training and evaluation                        |
| NB-05    | Step 2 — Background-removed training and evaluation                                 |
| NB-06    | Step 3 — Synthetic background training, evaluation, and cross-background robustness |
| NB-07    | Grand comparison — all steps and encoders                                           |

---

## Dataset

This project uses the
[LaboroTomato dataset](https://datasetninja.com/laboro-tomato),
a collection of 804 high-resolution greenhouse tomato images
annotated with polygon-based instance segmentation masks for
six ripeness classes across two tomato sizes.

**Note:** The original dataset test set was excluded from this
study due to misaligned polygon annotations. Only the 643 usable
images from the training portion were used, with a custom
stratified 70/15/15 split applied.

### Class definitions

| Class             | Description                           |
| ----------------- | ------------------------------------- |
| `b_fully_ripened` | Normal-size tomato, at least 90% red  |
| `b_half_ripened`  | Normal-size tomato, 30–89% red        |
| `b_green`         | Normal-size tomato, less than 30% red |
| `l_fully_ripened` | Cherry tomato, at least 90% red       |
| `l_half_ripened`  | Cherry tomato, 30–89% red             |
| `l_green`         | Cherry tomato, less than 30% red      |

### Expected directory structure

```text
/kaggle/input/datasets/sharifbek/laboro-tomato/tomato-dataset/
├── Train/
│   ├── img/        # Training images (.jpg)
│   └── ann/        # Supervisely JSON annotations
└── meta.json       # Dataset metadata and class definitions
```

If running locally, update `INPUT_DIR` in the configuration
cell of NB-01 to point to your dataset location.

---

## Setup Instructions

### Option 1 — Kaggle (Recommended)

The project was developed and executed on Kaggle using a Tesla T4
GPU. The simplest way to reproduce the results is to run the
notebook directly on Kaggle:

1. Open the notebook at:
   https://www.kaggle.com/code/sharifbek/tomato-quality-semantic-segmentation
2. Add the LaboroTomato dataset as an input
3. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
4. Run all cells

---

### Option 2 — Local Setup with Conda

```bash
# Create and activate environment
conda create -n tomato-seg python=3.9
conda activate tomato-seg

# Install PyTorch — choose your CUDA version
# CUDA 11.8:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# CPU only:
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt
```

---

### Option 3 — Local Setup with venv

```bash
python -m venv tomato-seg-env
source tomato-seg-env/bin/activate
# Windows: tomato-seg-env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Notebooks

### Jupyter

```bash
pip install jupyter
jupyter notebook
# Open and run notebooks in order: NB-01 through NB-07
```

### Papermill (automated execution)

```bash
pip install papermill

papermill tomato-segmentation.ipynb output.ipynb \
    -p INPUT_DIR /path/to/your/dataset \
    -p OUTPUT_DIR /path/to/output
```

### Convert to Python script

```bash
jupyter nbconvert --to python tomato-segmentation.ipynb
python tomato-segmentation.py
```

---

## Key Hyperparameters

| Parameter               | Value                        |
| ----------------------- | ---------------------------- |
| Image size              | 512 × 512 pixels             |
| Batch size (train)      | 8                            |
| Batch size (val/test)   | 4                            |
| Epochs (max)            | 40                           |
| Optimizer               | Adam                         |
| Learning rate           | 1e-4                         |
| Weight decay            | 1e-5                         |
| LR scheduler            | Cosine annealing             |
| Early stopping patience | 7 epochs                     |
| Loss function           | Weighted CE + Dice (α = 0.5) |
| Random seed             | 42                           |
| Hardware                | Tesla T4 GPU (Kaggle)        |

---

## Reproducibility

All experiments are fully reproducible:

- Random seeds fixed at 42 for Python, NumPy, PyTorch, and CUDA
- PyTorch deterministic mode enabled
  (`torch.backends.cudnn.deterministic = True`)
- Dataset split indices saved to disk and reused across all experiments
- Validation and test synthetic background assignments fixed with
  separate seeds (`SEED + 1000` and `SEED + 2000`) before training begins
- Both models initialized from identical ImageNet pretrained weights
  via the `segmentation-models-pytorch` library

> Do not modify seed settings if you want to reproduce the
> exact results reported in the thesis.

---

## Results Summary

| Condition          | Encoder         | Pixel Acc | Mean IoU | Mean Dice | ECE    |
| ------------------ | --------------- | --------- | -------- | --------- | ------ |
| Step 1 — Natural   | MobileNetV2     | 0.9406    | 0.6352   | 0.7693    | 0.0161 |
| Step 1 — Natural   | EfficientNet-B0 | 0.9333    | 0.6323   | 0.7670    | 0.0148 |
| Step 2 — Removed   | MobileNetV2     | 0.9717    | 0.7255   | 0.8334    | 0.0479 |
| Step 2 — Removed   | EfficientNet-B0 | 0.9783    | 0.7646   | 0.8599    | 0.0291 |
| Step 3 — Synthetic | MobileNetV2     | 0.9684    | 0.6995   | 0.8137    | 0.0284 |
| Step 3 — Synthetic | EfficientNet-B0 | 0.9732    | 0.7512   | 0.8514    | 0.0215 |

### Cross-background robustness

| Scenario                     | MobileNetV2 mIoU | EfficientNet-B0 mIoU |
| ---------------------------- | ---------------- | -------------------- |
| Natural → Natural (baseline) | 0.6352           | 0.6323               |
| (A) Natural → Synthetic      | 0.4678           | 0.3654               |
| (B) Synthetic → Natural      | 0.2690           | 0.2975               |
| (C) Synthetic → Synthetic    | 0.6995           | 0.7512               |

---

## Expected Execution Time

| Environment         | Approximate Time                     |
| ------------------- | ------------------------------------ |
| Kaggle Tesla T4 GPU | 8–10 hours (full pipeline)           |
| Local GPU           | varies by hardware                   |
| CPU only            | 12+ hours (development/testing only) |

---

## Outputs

Running the full pipeline produces the following outputs:

- `outputs/models/` — 6 model checkpoints (best weights per condition)
- `outputs/nb01/` — EDA figures and dataset split file
- `outputs/nb02/` — Background mode comparison figures, 200 synthetic backgrounds, class weights
- `outputs/nb04/` — Step 1 training curves, predictions, Grad-CAM, confusion matrices, calibration plots
- `outputs/nb05/` — Step 2 training curves, predictions, confusion matrices, calibration plots
- `outputs/nb06/` — Step 3 training curves, predictions, confusion matrices, calibration plots, robustness charts
- `outputs/nb07/` — Grand comparison figures and CSV tables

---

## Troubleshooting

| Issue                      | Solution                                       |
| -------------------------- | ---------------------------------------------- |
| CUDA out of memory         | Reduce `BATCH_SIZE_TRAIN` in NB-01 config cell |
| Dataset not found          | Verify `INPUT_DIR` path in NB-01 config cell   |
| Missing dependencies       | Run `pip install -r requirements.txt` again    |
| CuDNN determinism warnings | Safe to ignore — does not affect results       |
| Slow training on CPU       | Use GPU; CPU is only recommended for testing   |

---

## Academic Context

This project was implemented as part of a Master's thesis. The
experimental design, methodology, and analysis were developed to
systematically investigate background bias in tomato quality
classification using semantic segmentation. Full methodology,
results, and discussion are documented in the accompanying thesis.

---

## License

For academic use only. Dataset usage is subject to the
[LaboroTomato dataset license](https://datasetninja.com/laboro-tomato).
