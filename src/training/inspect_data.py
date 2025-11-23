"""Quick CLI to inspect the labels file and spot missing images."""

import argparse
import os

import pandas as pd


def inspect(data_dir: str = "data", filename_col: str = "image_name"):
    """Print dataset overview and report which sampled files are missing."""
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")

    df = pd.read_csv(labels_path)
    print("First rows of labels.csv:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("Total samples:", len(df))

    # Randomly sample a few entries to sanity-check file presence
    sample = df.sample(min(10, len(df)), random_state=42)
    missing = 0
    for _, row in sample.iterrows():
        filename = str(row[filename_col])
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print("Missing:", img_path)
            missing += 1
        else:
            print("OK:", img_path)

    print(f"\nMissing images in this sample: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--filename-col", default="image_name")
    args = parser.parse_args()

    inspect(
        data_dir=args.data_dir,
        filename_col=args.filename_col,
    )
