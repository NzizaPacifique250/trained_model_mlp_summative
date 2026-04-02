"""
FastAPI backend for the Cat vs Dog ML Pipeline.
Provides endpoints for prediction, data upload, retraining,
health monitoring, and dataset statistics.
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import json
import shutil
from typing import List
from src.prediction import predict_image
from src.model import retrain_model

app = FastAPI(title="ML Pipeline API")

# Allow React dev server and Docker frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ──
START_TIME = time.time()
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
HISTORY_PATH = "models/training_history.json"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


# ── Health / Uptime ──
@app.get("/health")
def health_check():
    """Returns API health status and uptime in seconds."""
    uptime = time.time() - START_TIME
    return {"status": "ok", "uptime_seconds": round(uptime, 2)}


# ── Prediction ──
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict Cat or Dog for a single uploaded image."""
    try:
        contents = await file.read()
        result = predict_image(contents)
        return {"filename": file.filename, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Data Upload ──
@app.post("/upload_data")
async def upload_bulk_data(
    label: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload multiple images for a specific label (cats / dogs)."""
    label = label.lower()
    if label not in ["cats", "dogs"]:
        raise HTTPException(status_code=400, detail="Label must be 'cats' or 'dogs'.")

    target_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(target_dir, exist_ok=True)

    saved_files = []
    for f in files:
        file_path = os.path.join(target_dir, f.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append(f.filename)

    return {
        "message": f"Uploaded {len(saved_files)} files to '{label}'.",
        "files": saved_files,
    }


# ── Dataset Statistics (for UI visualizations) ──
@app.get("/data/stats")
def dataset_stats():
    """Return image counts per class for train and validation splits."""
    stats = {}
    for split_name, split_dir in [("train", TRAIN_DIR), ("validation", VAL_DIR)]:
        stats[split_name] = {}
        for cls in ["cats", "dogs"]:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                count = len(
                    [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
                )
            else:
                count = 0
            stats[split_name][cls] = count
    return stats


# ── Training History (for UI visualizations) ──
@app.get("/model/history")
def training_history():
    """Return saved training history (accuracy / loss per epoch)."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="No training history found. Train the model first.")


# ── Retraining ──
def _background_retrain():
    try:
        print("Starting background retraining...")
        history = retrain_model(train_dir=TRAIN_DIR, validation_dir=VAL_DIR)
        # Persist updated training history
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(HISTORY_PATH, "w") as f:
            json.dump(history_dict, f, indent=2)
        print("Retraining completed successfully.")
    except Exception as e:
        print(f"Retraining error: {e}")


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining in the background."""
    background_tasks.add_task(_background_retrain)
    return {"message": "Retraining triggered in the background."}
