# MSENSIS-MLE-task - Cats vs Dogs Classification

Compact end-to-end ML project: data handling, Vision Transformer fine-tuning, FastAPI inference, and an optional Streamlit UI.

## Tech stack
- Hugging Face Transformers (ViT) on PyTorch
- NumPy and pandas for data handling
- FastAPI for the inference endpoint
- Streamlit for the optional UI
- Git for version control

## Directory Layout
```
MSENSIS-MLE-task/
|- data/             # (ignored) labels.csv and images/
|- models/           # (ignored) saved fine-tuned model/processor
|- src/
|  |- training/      # inspect_data.py, train_vit.py, eval_vit.py
|  |- inference/     # predict.py (CLI helper)
|  |- api/           # main.py (FastAPI /predict)
|  |- ui/            # app.py (Streamlit UI)
|- assets/           # screenshots (add your images here)
|- requirements.txt
|- .gitignore
`- README.md
```

## Installation
1. Clone or extract into `MSENSIS-MLE-task`.
2. (Recommended) Create and activate a virtual env:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # PowerShell
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Add your dataset under `data/`:
   - `data/labels.csv` with filename and label columns (defaults: `image_name`, `label`)
   - `data/images/` containing the referenced images

## Usage
- Inspect data paths:
  ```
  python src/training/inspect_data.py
  ```
- Train the model:
  ```
  python src/training/train_vit.py --data-dir data --output-dir models/vit_catsdogs
  ```
- Evaluate validation accuracy:
  ```
  python src/training/eval_vit.py --data-dir data --model-dir models/vit_catsdogs
  ```
- Run the API:
  ```
  uvicorn src.api.main:app --reload
  ```
- Launch the Streamlit UI:
  ```
  streamlit run src/ui/app.py
  ```
- CLI prediction:
  ```
  python src/inference/predict.py --image-path data/images/0.jpg
  ```

## Notes
- Defaults expect labels `cat` and `dog`; adjust flags if your CSV differs.
- The classifier head is reinitialized for 2 classes; the mismatched-size warning is expected.
- `data/`, `models/`, and `.venv/` are ignored to keep the repo lean.

## Screenshots
Add screenshots (e.g., Streamlit UI, API test) to `assets/` and reference them in this README, for example:
```
![Streamlit UI](assets/ui.png)
```

## GitHub remote
If you renamed the repo on GitHub, update the local remote:
```
git remote set-url origin https://github.com/<you>/MSENSIS-MLE-task.git
```
