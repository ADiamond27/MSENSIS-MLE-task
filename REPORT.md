# MSENSIS-MLE-task - Project Report

## Objective
Deliver a compact end-to-end pipeline to classify cat vs dog images: data prep, Vision Transformer fine-tuning, and serving predictions via API and UI. The goal is solid ML engineering across preprocessing, training, inference, and deployment.

## Dataset & Preprocessing
- Layout: `data/labels.csv` (`image_name`, `label`), images in `data/images/`.
- Safety: `filter_existing_images` drops missing or unreadable files and reports counts (~24,290 kept of 25,000).
- Labels: `cat -> 0`, `dog -> 1`; normalized to lowercase and trimmed.

## Vision Transformer & Model Choice
The Vision Transformer (ViT) splits an image into fixed patches, embeds them with positional encodings, and processes the sequence with a Transformer encoder. Self-attention captures global relationships better than purely local filters and adapts well from a pretrained checkpoint with limited task data. We chose `google/vit-base-patch16-224` for its size/accuracy balance and replaced the head with a 2-class layer (`ignore_mismatched_sizes=True`).

## Training Setup & Results
- Hyperparameters: `epochs=1`, `batch_size=8`, `learning_rate=2e-5`, `val_split=0.1`, `save_strategy="epoch"`.
- Processing: `ViTImageProcessor` handles resize to 224x224, normalization, tensor conversion.
- Results: Initial training showed ~93% validation accuracy. A full validation run achieved **99.96%** (2,428/2,429), and a 1,000-image sample measured ~99.8%:
  ```
  python src/training/eval_vit.py --data-dir data --model-dir models/vit_catsdogs
  ```
- Notes: Warnings about reinitialized classifier weights and `pin_memory` on CPU are expected and harmless.

## Inference & Serving
- CLI: `python src/inference/predict.py --image-path <path>` (loads from `models/vit_catsdogs`).
- API: FastAPI (`src/api/main.py`) exposes `/predict` for image uploads -> JSON.
- UI: Streamlit (`src/ui/app.py`) uploads an image, calls the API, and shows the prediction.

## Structure & Usage
- `src/training`: data inspection and ViT fine-tuning.
- `src/inference`: local prediction helper.
- `src/api`: FastAPI backend for real-time inference.
- `src/ui`: Streamlit front-end for manual testing.
- Ignored: `data/`, `models/`, `.venv/` (see `.gitignore`).

Run:
1. `python -m venv .venv` and activate it.
2. `pip install -r requirements.txt`.
3. Place dataset under `data/`.
4. Inspect: `python src/training/inspect_data.py`.
5. Train: `python src/training/train_vit.py --data-dir data --output-dir models/vit_catsdogs`.
6. API: `uvicorn src.api.main:app --reload`.
7. UI: `streamlit run src/ui/app.py`.
8. CLI: `python src/inference/predict.py --image-path data/images/0.jpg`.

## Observations & Future Work
- ViT adapts quickly; one epoch delivered strong accuracy.
- Modular layout simplifies testing and extension.
- Future: more epochs/tuning, augmentation, metrics logging, early stopping, model cards, tests, optional image-verification toggle for huge datasets.

## Conclusion
Built an end-to-end system: validated data, fine-tuned a ViT for cats vs dogs, served predictions via FastAPI, and provided a Streamlit UI. Key learnings: pretrained ViT adapts fast with clean data handling; clear separation of data/training/inference/API/UI keeps changes manageable. Next steps: richer evaluation (e.g., F1), broader tuning/augmentation, and automated tests.
