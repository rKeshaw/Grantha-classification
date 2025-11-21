# data_setup.py
# ==================================================
# Phase 0: Data Stratification
# Splits Raw data into Train (80%), Val (10%), Test (10%)
# ==================================================

import os
import shutil
import random
import config
from tqdm import tqdm

def stratified_split():
    print("🚀 Phase 0: Starting Stratified Split...")
    
    if not os.path.exists(config.RAW_DATA_DIR):
        print(f"❌ Error: Raw Data not found at {config.RAW_DATA_DIR}")
        return

    # Clean previous runs
    if os.path.exists(config.SPLIT_DATA_DIR):
        shutil.rmtree(config.SPLIT_DATA_DIR)

    # Define splits
    split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    classes = [d for d in os.listdir(config.RAW_DATA_DIR) if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))]
    print(f"Found {len(classes)} Grantha characters.")

    random.seed(42)
    
    for char_class in tqdm(classes, desc="Splitting Classes"):
        src_class_path = os.path.join(config.RAW_DATA_DIR, char_class)
        # inside stratified_split loop, change image gather:
        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp'))]
        if not images:
            print(f"Warning: class '{char_class}' contains no images; skipping.")
            continue

        
        # Random Shuffle
        random.shuffle(images)
        
        # Calculate indices
        n = len(images)
        train_end = int(n * split_ratios['train'])
        val_end = train_end + int(n * split_ratios['val'])
        
        splits = {
            'train': images[:train_end],
            'val':   images[train_end:val_end],
            'test':  images[val_end:]
        }

        # Move files
        for split, files in splits.items():
            dest_folder = os.path.join(config.SPLIT_DATA_DIR, split, char_class)
            os.makedirs(dest_folder, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src_class_path, f), os.path.join(dest_folder, f))

    print(f"✅ Data split complete. Output at: {config.SPLIT_DATA_DIR}")

if __name__ == "__main__":
    stratified_split()
