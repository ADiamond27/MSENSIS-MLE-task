import argparse
import os
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import ViTForImageClassification, ViTImageProcessor

DEFAULT_FILENAME_COL = "image_name"
DEFAULT_LABEL_COL = "label"

LABEL2ID = {"cat": 0, "dog": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def filter_existing_images(df: pd.DataFrame, images_dir: str, filename_col: str) -> pd.DataFrame:
    """Keep only rows whose image file actually exists on disk."""

    def exists_fn(fn: str) -> bool:
        img_path = os.path.join(images_dir, str(fn))
        return os.path.exists(img_path)

    mask = df[filename_col].apply(exists_fn)
    kept = mask.sum()
    total = len(df)
    print(f"Keeping {kept} samples out of {total} (dropped {total - kept} missing files).")
    return df[mask].reset_index(drop=True)


class CatDogEvalDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, filename_col: str, label_col: str) -> None:
        df = pd.read_csv(csv_path)
        df = filter_existing_images(df, images_dir, filename_col)
        self.data = df
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
            raise ValueError(f"Unknown label '{label_name}' (expected one of {list(LABEL2ID.keys())})")
        label_id = LABEL2ID[label_name]
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path).convert("RGB")
        return {"image": image, "label": label_id}


def evaluate(
    data_dir: str,
    model_dir: str,
    filename_col: str = DEFAULT_FILENAME_COL,
    label_col: str = DEFAULT_LABEL_COL,
    val_split: float = 0.1,
    batch_size: int = 16,
    subset: int = 0,
) -> None:
    csv_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")

    dataset = CatDogEvalDataset(csv_path, images_dir, filename_col, label_col)

    # Split off validation set (same logic as training)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if subset > 0 and subset < len(val_ds):
        print(f"Using subset of validation set: {subset} samples (out of {len(val_ds)})")
        indices = list(range(subset))
        val_ds = torch.utils.data.Subset(val_ds, indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    processor = ViTImageProcessor.from_pretrained(model_dir)
    model = ViTForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [b["image"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy on validation set: {acc:.4f} ({correct}/{total})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned ViT on cats vs dogs validation set")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with labels.csv and images/")
    parser.add_argument("--model-dir", type=str, default="models/vit_catsdogs", help="Directory of saved model")
    parser.add_argument("--filename-col", type=str, default=DEFAULT_FILENAME_COL)
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset", type=int, default=0, help="If >0 evaluate only on this many samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        val_split=args.val_split,
        batch_size=args.batch_size,
        subset=args.subset,
    )
