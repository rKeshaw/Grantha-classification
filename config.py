# config.py
"""Central configuration for Grantha-Net OCR project."""

from pathlib import Path
import os
import torch
import random
import numpy as np

# --- PATHS ---
DRIVE_ROOT = Path("./script_ds").resolve()

RAW_DATA_DIR = DRIVE_ROOT / "mainimage"
SPLIT_DATA_DIR = DRIVE_ROOT / "Split_Data"
CLEAN_DATA_DIR = DRIVE_ROOT / "Cleaned_Data"
AUG_DATA_DIR = DRIVE_ROOT / "Augmented_Data"
MODEL_SAVE_DIR = DRIVE_ROOT / "Saved_Models"
LOG_DIR = DRIVE_ROOT / "logs"

# --- IMAGE / PREPROCESSING ---
IMG_SIZE = 384              # square size expected by TrOCR-style models
UPSCALING_FACTOR = 3        # integer > 0
# Sauvola: must be odd and smaller than the (upscaled) image dimension
SAUVOLA_WINDOW = 85
SAUVOLA_K = 0.25
GRAYSCALE = True            # convert images to single channel before Sauvola

# --- AUGMENTATION ---
AUG_COPIES_PER_IMG = 15     # consider lowering if dataset is large
ROTATION_LIMIT = 7          # degrees; avoid flips for scripts
ALLOW_FLIP = False          # MUST be False for script OCR (do not flip)

# --- TRAINING HYPERPARAMETERS ---
TROCR_CHECKPOINT = "microsoft/trocr-small-handwritten"
VIT_CHECKPOINT   = "google/vit-base-patch16-384"
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 4e-5
USE_FP16 = True

# --- RUNTIME / REPRODUCIBILITY ---
RNG_SEED = 42
NUM_WORKERS = 4
SAVE_EVERY_N_EPOCHS = 1

# --- DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utilities ---
def setup_directories():
    for p in [RAW_DATA_DIR, SPLIT_DATA_DIR, CLEAN_DATA_DIR, AUG_DATA_DIR, MODEL_SAVE_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_config():
    assert IMG_SIZE > 0 and isinstance(IMG_SIZE, int)
    assert UPSCALING_FACTOR > 0 and isinstance(UPSCALING_FACTOR, int)
    assert isinstance(SAUVOLA_WINDOW, int) and SAUVOLA_WINDOW > 3, "SAUVOLA_WINDOW must be an int > 3"
    assert SAUVOLA_WINDOW % 2 == 1, "SAUVOLA_WINDOW should be odd for Sauvola implementation"
    upscaled_dim = IMG_SIZE * UPSCALING_FACTOR
    assert SAUVOLA_WINDOW < upscaled_dim, "SAUVOLA_WINDOW should be smaller than upscaled image dimension"
    assert BATCH_SIZE > 0 and isinstance(BATCH_SIZE, int)
    assert EPOCHS > 0 and isinstance(EPOCHS, int)
    assert AUG_COPIES_PER_IMG >= 0
    if ALLOW_FLIP:
        print("WARNING: ALLOW_FLIP=True. Flipping character images can break script semantics.")

# Run basic setup when executed
if __name__ == "__main__":
    setup_directories()
    validate_config()
    set_seed()
    print(f"Directories created under: {DRIVE_ROOT}")
    print(f"Using device: {DEVICE}")
    print("Config validated.")
