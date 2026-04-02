"""
Model definition, training, and retraining logic.
Uses MobileNetV2 with fine-tuning of the top convolutional layers
for improved accuracy on the Cat vs Dog classification task.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.preprocessing import get_data_generators, IMAGE_SIZE

HISTORY_PATH = "models/training_history.json"

# Number of layers to unfreeze from the top of MobileNetV2 during fine-tuning
FINE_TUNE_LAYERS = 30


def build_model():
    """
    Build MobileNetV2 with fine-tuned top layers and a custom classification head.

    Strategy:
    1. Load MobileNetV2 with ImageNet weights (no top).
    2. Freeze all layers initially.
    3. Unfreeze the last FINE_TUNE_LAYERS for fine-tuning.
    4. Add GlobalAveragePooling2D -> Dense(256) -> Dropout -> Dense(1, sigmoid).
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )

    # Freeze all base layers first
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    # The last FINE_TUNE_LAYERS remain trainable for fine-tuning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _get_callbacks(model_save_path):
    """Return a standard set of training callbacks."""
    return [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]


def _safe_steps(generator):
    return max(generator.samples // generator.batch_size, 1)


def _save_history(history):
    """Persist training history as JSON for the UI visualizations."""
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)


def train_model(train_dir, validation_dir, model_save_path="models/model.keras", epochs=15):
    """Train a new model from scratch with fine-tuning."""
    train_gen, val_gen, _ = get_data_generators(train_dir, validation_dir)
    model = build_model()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    history = model.fit(
        train_gen,
        steps_per_epoch=_safe_steps(train_gen),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=_safe_steps(val_gen),
        callbacks=_get_callbacks(model_save_path),
    )

    if not os.path.exists(model_save_path):
        model.save(model_save_path)

    _save_history(history)
    return history


def retrain_model(
    train_dir,
    validation_dir,
    existing_model_path="models/model.keras",
    new_model_path="models/model_retrained.keras",
    epochs=5,
):
    """Continue training an existing model with (potentially new) data."""
    if not os.path.exists(existing_model_path):
        print("No existing model found — training from scratch.")
        return train_model(train_dir, validation_dir, model_save_path=new_model_path, epochs=epochs)

    print(f"Loading existing model from {existing_model_path}")
    model = load_model(existing_model_path)

    train_gen, val_gen, _ = get_data_generators(train_dir, validation_dir)

    history = model.fit(
        train_gen,
        steps_per_epoch=_safe_steps(train_gen),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=_safe_steps(val_gen),
        callbacks=_get_callbacks(new_model_path),
    )

    if not os.path.exists(new_model_path):
        model.save(new_model_path)

    # Replace old model with retrained one
    try:
        import shutil
        shutil.copyfile(new_model_path, existing_model_path)
    except Exception as e:
        print(f"Could not replace original model: {e}")

    _save_history(history)
    return history
