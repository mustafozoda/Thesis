<div align="center">

# Evaluating Background Bias and Improving Model Reliability in Tomato Quality Classification Using Deep Learning and Explainable AI

</div>

## Project Overview

#### This project investigates how background information affects tomato quality classification models using semantic segmentation. The approach uses a three-step experiment with U-Net models (MobileNetV2 and EfficientNet-B0 encoders) to analyze background bias.

---

## Setup Instructions

### 1. Environment Setup

#### Using Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n tomato-seg python=3.9
conda activate tomato-seg

# Install PyTorch (choose appropriate CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio
```

#### Using venv

```bash
# Create virtual environment
python -m venv tomato-seg-env
source tomato-seg-env/bin/activate  # On Windows: tomato-seg-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

---

### 2. Install Dependencies

#### Install required packages:

```bash
pip install -r requirements.txt
```

---

### 3. Dataset Setup

#### The project uses the [LaboroTomato dataset](https://datasetninja.com/laboro-tomato). Place the dataset in the expected directory structure:

```text
/kaggle/input/datasets/sharifbek/tomato/tomato-dataset/
├── Train/
│   ├── img/          # Training images (.jpg)
│   └── ann/          # Training annotations (.json)
├── Test/
│   ├── img/          # Test images (.jpg)
│   └── ann/          # Test annotations (.json)
└── meta.json         # Dataset metadata
```

#### If your dataset is located elsewhere, update the INPUT_DIR path in the notebook's configuration cell.

## Running the Experiments

### Important: Fixed Seeds & Reproducibility

#### The notebook sets fixed seeds for reproducibility:

- `SEED = 42`

* Random seeds are set for Python's `random`, NumPy, and PyTorch (CPU and CUDA)

* CuDNN is set to deterministic mode

#### _Do not modify these settings if you want reproducible results._

---

## Runtime Settings

#### Default hyperparameters (configurable in the notebook):

- Image size: `512x512`

- Training epochs: `25`

- Batch size (train): `8`

- Batch size (test): `4`

- Learning rate: `1e-4`

* Early stopping patience: `5`

* Neutral background color (Step 2): `(128, 128, 128)`

---

## Running the Notebook

#### Option 1: Run with Jupyter

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook tomato-segmentation-final.ipynb

# Run all cells (Cell > Run All)
```

---

#### Option 2: Run with Papermill (for automated execution)

```bash
# Install papermill
pip install papermill

# Execute notebook
papermill tomato-segmentation-final.ipynb \
    tomato-segmentation-final-output.ipynb \
    -p INPUT_DIR /path/to/your/dataset \
    -p OUTPUT_DIR /path/to/output
```

---

#### Option 3: Convert to Python script and run

```bash
# Convert to Python script

jupyter nbconvert --to python tomato-segmentation-final.ipynb

# Run the script

python tomato-segmentation-final.py
```

## Expected Execution Time

- Full training (all steps) on a GPU: approximately 6-8 hours

- CPU-only execution: significantly longer (12+ hours recommended for testing only)

## Outputs

The notebook generates:

1. class_distribution.png - Visualization of class distribution

2. Model checkpoints (saved in OUTPUT_DIR)

3. Evaluation metrics and visualizations (Grad-CAM maps, reliability diagrams)

---

## Troubleshooting

### Common Issues

- CUDA out of memory: Reduce batch size in the configuration cell

- Dataset not found: Verify INPUT_DIR path matches your dataset location

- Missing dependencies: Run pip install -r requirements.txt again

- Reproducibility warnings: Ignore CuDNN determinism warnings; they don't affect results

---

## Execution Environment

> **Note:** This project was developed and executed in a [Kaggle environment](https://www.kaggle.com/code/sharifbek/tomato-segmentation-final). The directory structure, default `INPUT_DIR` path, and GPU configuration assume a Kaggle Notebook runtime. If you are running the project locally or on another platform, make sure to adjust dataset paths, hardware settings, and environment configurations accordingly.

## Academic Context

This project was implemented as part of a Master's thesis research project. The experimental design, methodology, and analysis were developed to systematically investigate background bias in tomato quality classification using semantic segmentation models.

### Getting Help

### _Check the notebook's comments and markdown cells for detailed explanations of each step._

---

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/sharifbek/tomato-segmentation-final)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/mustafozoda/Thesis)

</div>
