"""Utility helpers for offline single-image predictions."""

import argparse
import os
from typing import Union

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Default location of the fine-tuned weights and processor
MODEL_DIR = os.path.join("models", "vit_catsdogs")
# Prefer GPU when available for inference speed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str = MODEL_DIR):
    """Load a fine-tuned ViT model and its processor from disk."""
    processor = ViTImageProcessor.from_pretrained(model_dir)
    model = ViTForImageClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return model, processor


def predict_image(image_input: Union[str, Image.Image], model, processor):
    """Run a single-image prediction from a filepath or PIL Image instance."""
    if isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.open(image_input).convert("RGB")

    # Preprocess image and execute the model
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        conf = float(probs[pred_id])
    label = model.config.id2label[pred_id]
    return label, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image to classify")
    args = parser.parse_args()

    # Load artifacts once, then classify the requested image
    model, processor = load_model(MODEL_DIR)
    label, conf = predict_image(args.image_path, model, processor)
    print(f"Prediction: {label} (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
