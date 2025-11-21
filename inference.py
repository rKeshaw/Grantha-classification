# inference.py
# ==================================================
# Inference / Demo Script for Grantha Classifier
# Loads the trained ViT classifier and predicts on a raw image.
# ==================================================

import os
import sys
import cv2
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from utils_img import clean_palm_leaf
import config


def predict_grantha_class(image_path: str, model_path: str, top_k: int = 1):
    """
    Run inference on a single Grantha character image using a trained ViT classifier.

    Parameters
    ----------
    image_path : str
        Path to the raw input image (palm leaf character crop).
    model_path : str
        Path to the trained model directory (e.g., config.MODEL_SAVE_DIR/final_grantha_classifier).
    top_k : int, optional
        Number of top predictions to return. Default is 1.

    Returns
    -------
    predictions : list[tuple[str, float]]
        A list of (label, probability) pairs sorted by probability descending.
    clean_img_rgb : np.ndarray
        The cleaned RGB image that was fed to the model (for debugging/visualization).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load processor + model from the trained directory
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    model.eval()

    # 2. Load raw image
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    # 3. Apply Digital Scribe cleaning (matches training domain)
    clean_img = clean_palm_leaf(raw_img)
    if clean_img is None:
        raise ValueError("clean_palm_leaf returned None (image may be invalid).")

    # clean_img is typically grayscale -> convert to RGB
    if clean_img.ndim == 2:
        clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)
    else:
        clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

    # 4. Preprocess for ViT
    inputs = image_processor(images=clean_img_rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # 5. Predict
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits  # shape: [1, num_labels]
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: [num_labels]

    # Map indices back to labels
    id2label = model.config.id2label
    num_labels = logits.shape[-1]

    top_k = max(1, min(top_k, num_labels))
    topk_probs, topk_indices = torch.topk(probs, k=top_k)

    predictions = []
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        label = id2label.get(idx, str(idx))
        predictions.append((label, prob))

    return predictions, clean_img_rgb


if __name__ == "__main__":
    # Default model path: last saved classifier
    MODEL_PATH = os.path.join(config.MODEL_SAVE_DIR, "final_grantha_classifier")

    # CLI usage: python inference.py path/to/image.png [top_k]
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.png [top_k]")
        sys.exit(1)

    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    if not os.path.exists(MODEL_PATH):
        print(f"Model path not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"Image path not found: {image_path}")
        sys.exit(1)

    print("🔮 Predicting class...")
    preds, processed_img = predict_grantha_class(image_path, MODEL_PATH, top_k=top_k)

    if top_k == 1:
        label, prob = preds[0]
        print(f"📜 Predicted Class: {label} (p={prob:.4f})")
    else:
        print("📜 Top predictions:")
        for label, prob in preds:
            print(f"  - {label}: {prob:.4f}")

    # Save what the model saw (debug)
    debug_path = "debug_model_view.png"
    # Note: processed_img is RGB; OpenCV expects BGR for saving
    debug_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(debug_path, debug_bgr)
    print(f"Saved debug image to {debug_path}")
