# train.py
# ==================================================
# Phase 3: Training an Image Classifier (ViT)
# Fine-tunes google/vit-base-patch16-384 on Grantha.
# ==================================================

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import config

# --- 1. Metadata Generation Helper ---
def create_metadata_csv(split_name: str) -> pd.DataFrame:
    base_path = os.path.join(config.AUG_DATA_DIR, split_name)
    if not os.path.exists(base_path):
        print(f"Info: split folder not found: {base_path}")
        return pd.DataFrame(columns=["file_name", "text"])
    data = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for img in sorted(os.listdir(folder_path)):
            if img.startswith("."):
                continue
            if not img.lower().endswith((".tiff", ".tif", ".jpg", ".jpeg", ".png", ".bmp")):
                continue
            data.append({"file_name": os.path.join(folder_path, img), "text": folder})
    return pd.DataFrame(data)


# --- 2. Custom Dataset Class ---
class GranthaClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_processor, label2id: dict):
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        label_str = row["text"]

        image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image at {file_name}")

        # Ensure 3-channel RGB for processor; if grayscale, up-convert
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processed = self.image_processor(image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        if label_str not in self.label2id:
            raise KeyError(f"Label '{label_str}' not found in label2id mapping.")

        label_id = self.label2id[label_str]

        return {"pixel_values": pixel_values, "labels": torch.tensor(label_id, dtype=torch.long)}


# --- 3. Metrics: Accuracy ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean().item()
    return {"accuracy": accuracy}


# --- 4. Main Training Logic ---
if __name__ == "__main__":
    print("Phase 3: Initializing ViT Classification Training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("FP16 enabled in config:", getattr(config, "USE_FP16", False))

    # Load DataFrames
    train_df = create_metadata_csv("train")
    val_df = create_metadata_csv("val")
    test_df = create_metadata_csv("test")

    print(f"Training Samples:   {len(train_df)}")
    print(f"Validation Samples: {len(val_df)}")
    print(f"Test Samples:       {len(test_df)}")

    if len(train_df) == 0:
        raise SystemExit("No training samples found — check AUG_DATA_DIR/train")

    # Build label mappings from TRAIN split
    classes = sorted(train_df["text"].unique())
    label2id = {label: idx for idx, label in enumerate(classes)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(classes)

    print(f"Number of classes: {num_labels}")

    # Load image processor + model (validate checkpoint compatibility)
    # try:
    image_processor = AutoImageProcessor.from_pretrained(config.VIT_CHECKPOINT)
    model = ViTForImageClassification.from_pretrained(
        config.VIT_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)
    # except Exception as e:
    #     raise RuntimeError(
    #         f"Failed to load ViT from checkpoint '{config.VIT_CHECKPOINT}'. "
    #         "Ensure MODEL_CHECKPOINT points to a ViT (e.g., 'google/vit-base-patch16-384') or change your config."
    #     ) from e

    # Create Datasets
    train_dataset = GranthaClassificationDataset(train_df, image_processor, label2id)
    val_dataset = GranthaClassificationDataset(val_df, image_processor, label2id)
    test_dataset = GranthaClassificationDataset(test_df, image_processor, label2id)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config.MODEL_SAVE_DIR,
        save_strategy="epoch",
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        fp16=config.USE_FP16,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=getattr(config, "NUM_WORKERS", 4),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    print("✅ Training complete.")

    # Evaluate
    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("Validation metrics:", val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Test metrics:", test_metrics)

    final_path = os.path.join(config.MODEL_SAVE_DIR, "final_grantha_classifier")
    trainer.save_model(final_path)
    image_processor.save_pretrained(final_path)
    print("✅ Final classifier saved at:", final_path)
