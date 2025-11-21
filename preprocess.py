# preprocess.py
# ==================================================
# Phase 1: Mass Pre-processing
# Applies utils_img.clean_palm_leaf to the dataset
# ==================================================

import os
import cv2
import config
from utils_img import clean_palm_leaf
from tqdm import tqdm

def run_preprocessing():
    print("Phase 1: Starting 'Digital Scribe' Restoration...")
    splits = ['train', 'val', 'test']

    for split in splits:
        src_root = os.path.join(config.SPLIT_DATA_DIR, split)
        dst_root = os.path.join(config.CLEAN_DATA_DIR, split)

        if not os.path.exists(src_root):
            print(f"Info: source split folder missing: {src_root} — skipping.")
            continue

        classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
        print(f"Processing {split} set ({len(classes)} classes)...")
        for char_class in tqdm(classes):
            src_folder = os.path.join(src_root, char_class)
            dst_folder = os.path.join(dst_root, char_class)
            os.makedirs(dst_folder, exist_ok=True)

            images = [f for f in os.listdir(src_folder)
                      if f.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp'))]
            for img_name in images:
                img_path = os.path.join(src_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Warning: unable to read {img_path}; skipping.")
                    continue
                try:
                    clean_img = clean_palm_leaf(img)
                    if clean_img is not None:
                        new_name = os.path.splitext(img_name)[0] + ".png"
                        cv2.imwrite(os.path.join(dst_folder, new_name), clean_img)
                except Exception as e:
                    print(f"Skipped {img_path}: {repr(e)}")

    print(f"✅ Restoration complete. Output at: {config.CLEAN_DATA_DIR}")

if __name__ == "__main__":
    run_preprocessing()
