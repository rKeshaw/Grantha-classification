# augment.py
# ==================================================
# Phase 2: Elastic Augmentation (Synthetic Data Gen)
# Uses Albumentations to simulate handwriting physics.
# Robusted version.
# ==================================================

import os
import cv2
import shutil
import numpy as np
import albumentations as A
import config
from tqdm import tqdm

# ---------- Custom morphological transforms ----------
class MorphologicalDilation(A.ImageOnlyTransform):
    def __init__(self, ksize=3, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.ksize = ksize

    def apply(self, image, **params):
        kernel = np.ones((self.ksize, self.ksize), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)


class MorphologicalErosion(A.ImageOnlyTransform):
    def __init__(self, ksize=3, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.ksize = ksize

    def apply(self, image, **params):
        kernel = np.ones((self.ksize, self.ksize), np.uint8)
        return cv2.erode(image, kernel, iterations=1)


def get_scribe_transforms():
    """Defines the elastic deformation pipeline."""
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=config.ROTATION_LIMIT,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.8,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        alpha_affine=10,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=255,
                        p=1.0,
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.3,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=255,
                        p=1.0,
                    ),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    MorphologicalDilation(ksize=3, p=1),
                    MorphologicalErosion(ksize=3, p=1),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=3,
                max_height=20,
                max_width=20,
                fill_value=255,
                p=0.2,
            ),
        ]
    )


def _is_image_file(name: str) -> bool:
    return name.lower().endswith((".tiff", ".tif", ".jpg", ".jpeg", ".png", ".bmp"))


def run_augmentation():
    print("Phase 2: Generating Synthetic Samples...")

    # Ensure source exists
    for split in ["val", "test"]:
        src = os.path.join(config.CLEAN_DATA_DIR, split)
        dst = os.path.join(config.AUG_DATA_DIR, split)
        if not os.path.exists(src):
            print(f"Info: source split missing: {src}. Skipping copy for {split}.")
            continue
        # Remove previous destination, then copy
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {split} => {dst}")

    # Augment train only if present
    train_src = os.path.join(config.CLEAN_DATA_DIR, "train")
    train_dst = os.path.join(config.AUG_DATA_DIR, "train")
    if not os.path.exists(train_src):
        print(f"Error: training source not found at {train_src}. Aborting augmentation.")
        return
    os.makedirs(train_dst, exist_ok=True)

    transform = get_scribe_transforms()

    classes = [d for d in os.listdir(train_src) if os.path.isdir(os.path.join(train_src, d))]
    print(f"Found {len(classes)} classes in training source.")

    # Safety: avoid runaway dataset growth
    if config.AUG_COPIES_PER_IMG > 50:
        print(f"Warning: AUG_COPIES_PER_IMG = {config.AUG_COPIES_PER_IMG} (very large).")

    for char_class in tqdm(classes, desc="Augmenting classes"):
        src_folder = os.path.join(train_src, char_class)
        dst_folder = os.path.join(train_dst, char_class)
        os.makedirs(dst_folder, exist_ok=True)

        image_files = [f for f in os.listdir(src_folder) if _is_image_file(f)]
        for img_name in image_files:
            img_path = os.path.join(src_folder, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: failed to read {img_path}; skipping.")
                continue

            # Normalize single-channel to 3-channel for safer transforms 
            if image.ndim == 2:
                image_for_aug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_for_aug = image

            # Save original 
            orig_name = f"orig_{img_name}"
            cv2.imwrite(os.path.join(dst_folder, orig_name), image)

            # Generate clones
            for i in range(config.AUG_COPIES_PER_IMG):
                aug = transform(image=image_for_aug)["image"]
                # If original was single-channel, convert back to grayscale to save space
                if image.ndim == 2:
                    aug_gray = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
                    out_name = f"aug_{i}_{img_name}"
                    cv2.imwrite(os.path.join(dst_folder, out_name), aug_gray)
                else:
                    out_name = f"aug_{i}_{img_name}"
                    cv2.imwrite(os.path.join(dst_folder, out_name), aug)

    print(f"✅ Augmentation complete. Dataset ready at: {config.AUG_DATA_DIR}")


if __name__ == "__main__":
    run_augmentation()
