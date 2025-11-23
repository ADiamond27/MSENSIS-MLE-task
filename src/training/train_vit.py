"""
Fine-tune a Vision Transformer for the animals task (cats vs dogs classification).

This script demonstrates how to perform transfer learning using Hugging Face
Transformers and PyTorch. It reads a CSV file of image filenames and labels
(cat or dog), splits the data into training and validation sets, applies
preprocessing appropriate for a Vision Transformer, trains for a configurable
number of epochs, and saves the resulting model and image processor to disk.
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, random_split
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)

# Default column names in labels.csv
# Set the default filename column to ``image_name`` because the provided dataset
# uses ``image_name`` and ``label``. Override with --filename-col if your CSV
# uses a different field for the image name.
DEFAULT_FILENAME_COL = "image_name"
DEFAULT_LABEL_COL = "label"

# Map class names to integer IDs and back
LABEL2ID = {"cat": 0, "dog": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def filter_existing_images(df: pd.DataFrame, images_dir: str, filename_col: str) -> pd.DataFrame:
    """Filter to rows whose image files exist and can be opened."""
    valid_mask: List[bool] = []
    missing = 0
    unreadable = 0

    for fn in df[filename_col]:
        img_path = os.path.join(images_dir, str(fn))
        if not os.path.exists(img_path):
            missing += 1
            valid_mask.append(False)
            continue
        try:
            # verify the file is readable to avoid runtime crashes during training
            with Image.open(img_path) as img:
                img.verify()
            valid_mask.append(True)
        except (UnidentifiedImageError, OSError):
            unreadable += 1
            valid_mask.append(False)

    mask = pd.Series(valid_mask, index=df.index)
    kept = mask.sum()
    total = len(df)
    print(
        f"Keeping {kept} samples out of {total} "
        f"(missing files: {missing}, unreadable images: {unreadable})."
    )
    return df[mask].reset_index(drop=True)


class CatDogDataset(Dataset):
    """Dataset for loading images and labels from a CSV file."""

    def __init__(self, csv_path: str, images_dir: str, filename_col: str, label_col: str) -> None:
        df = pd.read_csv(csv_path)
        df = filter_existing_images(df, images_dir, filename_col)

        self.data = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.filename_col = filename_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        filename = row[self.filename_col]
        label_name = str(row[self.label_col]).strip().lower()
        if label_name not in LABEL2ID:
            raise ValueError(f"Unknown label '{label_name}' in dataset; expected one of {list(LABEL2ID.keys())}")
        label_id = LABEL2ID[label_name]
        img_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Image at '{img_path}' is unreadable") from exc
        return {"image": image, "label": label_id}


def collate_fn(batch: List[Dict[str, Any]], processor: ViTImageProcessor) -> Dict[str, Any]:
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs


def train(
    data_dir: str,
    output_dir: str,
    filename_col: str = DEFAULT_FILENAME_COL,
    label_col: str = DEFAULT_LABEL_COL,
    val_split: float = 0.1,
    epochs: int = 1,
    batch_size: int = 8,
    model_name: str = "google/vit-base-patch16-224-in21k",
) -> None:
    csv_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")
    dataset = CatDogDataset(csv_path, images_dir, filename_col, label_col)

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    processor = ViTImageProcessor.from_pretrained(model_name)
    # ignore_mismatched_sizes ensures the classifier head is re-initialised for our 2 labels
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Custom collate function to apply the image processor
    def collate(batch: List[Dict[str, Any]]):
        return collate_fn(batch, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # renamed in transformers 4.57+
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=2e-5,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        tokenizer=processor,
    )

    trainer.train()

    # Save the fine-tuned model and processor
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ViT on the animals task (cats vs dogs)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing labels.csv and images/")
    parser.add_argument("--output-dir", type=str, default="models/vit_catsdogs", help="Directory to store the trained model and processor")
    parser.add_argument("--filename-col", type=str, default=DEFAULT_FILENAME_COL, help="Column name in labels.csv for image filenames")
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL, help="Column name in labels.csv for class labels")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data reserved for validation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224", help="Name of the pre-trained model to fine-tune")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
