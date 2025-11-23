import io
import os

import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from src.inference.predict import predict_image

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join("models", "vit_catsdogs"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model once at startup so each request can reuse them
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Simple FastAPI app exposing a single prediction endpoint
app = FastAPI(title="Cats vs Dogs API")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image bytes and normalise to RGB
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess and run the model
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        conf = float(probs[pred_id])
    label = model.config.id2label[pred_id]
    return {"label": label, "confidence": conf}
