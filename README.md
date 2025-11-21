# Grantha-Net — OCR & Classification Pipeline for Grantha Script

**Grantha-Net** is a modular pipeline for preparing, augmenting, and classifying Grantha characters (palm-leaf manuscript glyphs). It implements a domain-aware preprocessing (“Digital Scribe”), script-sensitive augmentations, and fine-tuning of a Vision Transformer (ViT) for per-character recognition. The repository also contains an inference utility for single-image prediction.

This README documents repository structure, installation, per-phase usage, configuration, best practices, troubleshooting, and recommendations for production use.

---

# Table of contents

* Project overview
* Repository layout
* Quick installation
* Configuration (important variables in `config.py`)
* Expected input dataset layout
* Phase-by-phase quickstart

  * Phase 0 — Stratified split (`data_setup.py`)
  * Phase 1 — Preprocessing / Digital Scribe (`preprocess.py`, `utils_img.py`)
  * Phase 2 — Augmentation (`augment.py`)
  * Phase 3 — Training (`train.py`)
  * Inference (`inference.py`)
* Metrics and evaluation checklist
* Reproducibility & performance tips
* Common issues & troubleshooting
* Recommended production changes
* License, contribution, and contact
* Quick command summary

---

# Project overview

Grantha-Net prepares raw palm-leaf character images, restores stroke detail with a targeted preprocessing pipeline, synthesizes handwriting-like variability, and fine-tunes a ViT backbone for classification. The code is intentionally modular: each phase is standalone and can be reused or replaced.

Primary goals:

* Robust restoration for palm-leaf degradation (fading, stains, lighting)
* Augmentations that respect script orientation and stroke semantics
* Simple, reproducible training pipeline using Hugging Face `transformers` Trainer API
* Lightweight inference utility for single images

---

# Repository layout (key files)

```
.
├── config.py                # Central configuration (paths, hyperparams, flags)
├── data_setup.py            # Phase 0: stratified split
├── preprocess.py            # Phase 1: Digital Scribe (calls utils_img)
├── utils_img.py             # Palm-leaf restoration (Sauvola, upscaling, blob filtering)
├── augment.py               # Phase 2: Albumentations-based augmentation
├── train.py                 # Phase 3: ViT classifier training and evaluation
├── inference.py             # Predict a class for a single image
├── requirements.txt         # (recommended) Python dependencies
└── README.md
```

**Note:** `script_ds/` (dataset and artifacts) is expected at runtime but should be added to `.gitignore` (see Recommended production changes).

---

# Quick installation

Recommended Python: **3.10+**.

Install dependencies (example; prefer pinned versions in `requirements.txt`):

```bash
pip install -r requirements.txt
```

Minimal package set (example):

```
torch
torchvision
transformers
albumentations==1.3.1
opencv-python-headless
scikit-image
pandas
tqdm
numpy
scikit-learn
evaluate
accelerate
```

**GPU:** install `torch` that matches your CUDA version (see PyTorch installation instructions). If you intend to visualize locally, use `opencv-python` instead of `opencv-python-headless`.

---

# Configuration (`config.py`)

All important paths and hyperparameters are centralized in `config.py`. Before running, review and tune:

* `DRIVE_ROOT` (default: `./script_ds`)
* `RAW_DATA_DIR` (default: `DRIVE_ROOT/mainimage`)
* `SPLIT_DATA_DIR`, `CLEAN_DATA_DIR`, `AUG_DATA_DIR`, `MODEL_SAVE_DIR`
* `IMG_SIZE` — input resolution expected by model (default `384` for `vit-base-patch16-384`)
* `UPSCALING_FACTOR` — integer used by preprocessing upscaling
* `SAUVOLA_WINDOW`, `SAUVOLA_K` — Sauvola binarization parameters (validated by code)
* `AUG_COPIES_PER_IMG` — synthetic copies per training image (default 15; tune for dataset size)
* `ROTATION_LIMIT` and `ALLOW_FLIP` — DO NOT enable flips for script images (orientation sensitive)
* `VIT_CHECKPOINT` — ViT backbone (recommended `"google/vit-base-patch16-384"`)
* `TROCR_CHECKPOINT` — (optional) checkpoint for TrOCR based workflows
* Training hyperparameters: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `USE_FP16`
* `RNG_SEED`, `NUM_WORKERS`, `DEVICE`

Run basic setup and validation:

```bash
python -c "import config; config.setup_directories(); config.validate_config(); config.set_seed()"
```

---

# Expected input dataset layout

Place raw glyph crops in a per-class directory structure:

```
script_ds/mainimage/
  class_A/
    img001.tif
    img002.jpg
    ...
  class_B/
    ...
```

Supported image formats: `.tiff, .tif, .jpg, .jpeg, .png, .bmp`.

Each folder name is treated as the ground-truth label.

---

# Phase-by-phase quickstart

Run commands from the repository root in the environment where `config.DRIVE_ROOT` is accessible.

## Phase 0 — Stratified split

Creates 80/10/10 train/val/test splits, preserving class distribution.

Command:

```bash
python data_setup.py
```

Output: `config.SPLIT_DATA_DIR` (e.g. `script_ds/Split_Data/`)

Notes:

* The script **copies** files (raw data remains intact).
* Very small classes may produce empty val/test sets; inspect class counts.

## Phase 1 — Preprocessing / Digital Scribe

Restores and binarizes images with a domain-aware pipeline.

Command:

```bash
python preprocess.py
```

Output: `config.CLEAN_DATA_DIR` (e.g. `script_ds/Cleaned_Data/`)

Important:

* `clean_palm_leaf` returns an 8-bit grayscale image with **black text = 0** and **white background = 255**.
* Sauvola window is auto-validated to be odd and smaller than the image dimension.

## Phase 2 — Augmentation

Generates handwriting-style and damage augmentations for training.

Command:

```bash
python augment.py
```

Output: `config.AUG_DATA_DIR` (e.g. `script_ds/Augmented_Data/`)

Behaviour:

* `val` and `test` are copied unmodified from `CLEAN_DATA_DIR`.
* `train` is augmented and saved with `orig_` and `aug_{i}_` prefixes.
* Augmentation pipeline: `ShiftScaleRotate`, `ElasticTransform` / `GridDistortion`, morphological erosion/dilation, `CoarseDropout`.
* Single-channel images are temporarily converted to 3-channel for transforms and saved back as grayscale when appropriate.

Safety:

* Avoid extremely large `AUG_COPIES_PER_IMG`; a warning prints when > 50.

## Phase 3 — Training (ViT classifier)

Fine-tunes a Vision Transformer for character classification using HF Trainer.

Command:

```bash
python train.py
```

Output:

* Checkpoints and logs in `config.MODEL_SAVE_DIR`.
* Final model in `config.MODEL_SAVE_DIR/final_grantha_classifier/` (model + processor saved).

Notes:

* The script builds metadata from `AUG_DATA_DIR/<split>/<class>`.
* It replaces the classifier head to `num_labels` and uses `ignore_mismatched_sizes=True` when loading to reuse backbone weights.
* To enable evaluation during training, set `evaluation_strategy="epoch"` in the `TrainingArguments` block in `train.py`.

## Inference demo

Predict a class for a single raw image.

Command:

```bash
python inference.py path/to/image.png [top_k]
```

Example:

```bash
python inference.py script_ds/mainimage/class_A/img123.png 3
```

Output:

* Top-k label(s) and probabilities printed.
* Saved debug image `debug_model_view.png` showing the processed input.


---

# Quick command summary

```bash
# 0. Prepare environment & config
python -c "import config; config.setup_directories(); config.validate_config(); config.set_seed()"

# 1. Create stratified splits
python data_setup.py

# 2. Preprocess / Digital Scribe restoration
python preprocess.py

# 3. Generate augmentations
python augment.py

# 4. Train ViT classifier
python train.py

# 5. Inference on a single image (top_k optional)
python inference.py path/to/image.png [top_k]
```
