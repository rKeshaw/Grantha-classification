# utils_img.py
# ==================================================
# The "Digital Scribe" Module
# Contains specific algorithms for Palm Leaf Restoration.
# Robusted and corrected version.
# ==================================================

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage import measure
import config

def _safe_sauvola_window(img_shape, requested):
    """Return a safe odd Sauvola window less than min(img_shape)."""
    min_dim = min(img_shape)
    win = int(requested)
    if win >= min_dim:
        win = min_dim - 1
    if win % 2 == 0:
        win = max(3, win - 1)
    # Final safety
    return max(3, win)

def clean_palm_leaf(img):
    """
    Restoration Pipeline (robust):
      1. Accepts color or grayscale.
      2. Extracts red channel (if color) or uses grayscale.
      3. Background estimation & normalization.
      4. Bicubic upscaling.
      5. Unsharp masking with safe dtype handling.
      6. Sauvola binarization with validated window.
      7. Blob filtering with area thresholds scaled to image size.
      8. Morphological closing.
    Returns uint8 image with BLACK text (0) on WHITE background (255).
    """
    if img is None:
        return None

    # If grayscale, keep it; if color, use the red channel as original pipeline did.
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        gray = r
    else:
        # Unexpected channel count: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Background estimation
    dilated_bg = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_blur = cv2.medianBlur(dilated_bg, 21)

    # Normalize: (Signal / Background) style normalization, then scale
    norm_img = 255 - cv2.absdiff(gray, bg_blur)
    norm_img = cv2.normalize(
        norm_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 3. Upscaling
    scale = max(1, int(config.UPSCALING_FACTOR))
    img_up = cv2.resize(norm_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 4. Unsharp mask / sharpening (avoid overflow)
    # Use int16 for convolution then clip back to uint8
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.int16)
    img_int = img_up.astype(np.int16)
    img_sharp = cv2.filter2D(img_int, ddepth=-1, kernel=kernel_sharpen)
    img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)

    # 5. Sauvola Binarization with safe window
    win = _safe_sauvola_window(img_sharp.shape, config.SAUVOLA_WINDOW)
    thresh = threshold_sauvola(img_sharp, window_size=win, k=config.SAUVOLA_K)
    binary = (img_sharp > thresh).astype(np.uint8) * 255  # 0 or 255, uint8

    # Ensure final convention: BLACK text (0) on WHITE background (255)
    # Determine which value is background: the majority pixel value is background.
    n_white = int((binary == 255).sum())
    n_black = int((binary == 0).sum())
    if n_white < n_black:
        # More black than white => background is (wrongly) black => invert
        binary = cv2.bitwise_not(binary)
        n_white, n_black = n_black, n_white  # swap for consistency

    # 6. Blob filtering: treat text = black (0). Label black pixels as foreground objects.
    # Convert to boolean mask for labeling (foreground True)
    foreground = (binary == 0)

    # label connected components
    labels = measure.label(foreground, connectivity=2)
    mask = np.zeros(binary.shape, dtype=np.uint8)

    # scale area thresholds relative to image size (avoid hard-coded constants)
    img_area = binary.shape[0] * binary.shape[1]
    min_area = max(5, int(0.0005 * img_area))   # e.g. for 384x384 => ~73
    max_area = max(100, int(0.2 * img_area))    # avoids keeping massive stains

    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        label_mask = (labels == lbl).astype(np.uint8) * 255
        num_pixels = int(cv2.countNonZero(label_mask))
        if min_area <= num_pixels <= max_area:
            mask = cv2.bitwise_or(mask, label_mask)

    # 7. Morphological closing to connect broken strokes
    morph_kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

    # closed currently has foreground as 255; convert that to black-on-white convention:
    # foreground 255 -> text; we want text = 0, background = 255
    final = cv2.bitwise_not(closed)

    return final
