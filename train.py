"""
Standalone Model Training Script
Trains a MobileNetV2-based Cat vs Dog classifier and saves
the model + training history for use by the API and UI.

Usage:
    python train.py
"""

import os
import json
from src.model import train_model

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
MODEL_PATH = "models/model.keras"
HISTORY_PATH = "models/training_history.json"
EPOCHS = 15


def main():
    # Verify data directories exist and have content
    for d in [TRAIN_DIR, VAL_DIR]:
        if not os.path.exists(d):
            print(f"Error: {d} not found. Run 'python download_data.py' first.")
            return
        subdirs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
        if not subdirs:
            print(f"Error: {d} has no class subdirectories. Run 'python download_data.py' first.")
            return

    os.makedirs("models", exist_ok=True)

    print(f"Training model for {EPOCHS} epochs...")
    print(f"  Train data:      {TRAIN_DIR}")
    print(f"  Validation data:  {VAL_DIR}")
    print(f"  Model output:    {MODEL_PATH}")

    history = train_model(
        train_dir=TRAIN_DIR,
        validation_dir=VAL_DIR,
        model_save_path=MODEL_PATH,
        epochs=EPOCHS,
    )

    # Save training history as JSON for the UI visualizations
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nModel saved to:   {MODEL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print("Training complete!")


if __name__ == "__main__":
    main()
