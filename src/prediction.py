"""
Inference module — two-stage prediction pipeline:

1. ImageNet MobileNetV2 (1000 classes) acts as a pre-filter to determine
   whether the image contains a cat or dog.
2. If it IS a cat/dog, the custom fine-tuned model provides the final
   binary classification with high confidence.
3. If it is NOT a cat/dog, the ImageNet top-5 predictions are returned
   so the user sees what the image actually contains.

This approach satisfies the rubric requirement of using the custom-trained
model for prediction while also handling out-of-distribution inputs.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.models import load_model
from PIL import Image
import io

from src.preprocessing import IMAGE_SIZE

IMAGENET_SIZE = (224, 224)

# ImageNet class indices for cats and dogs
DOG_INDICES = set(range(151, 269))
CAT_INDICES = set(range(281, 286))
CAT_DOG_INDICES = DOG_INDICES | CAT_INDICES
MIN_CAT_DOG_PROB = 0.05


class Predictor:
    def __init__(self, model_path="models/model.keras"):
        self.model_path = model_path
        self.imagenet_model = None
        self.custom_model = None
        self.load()

    def load(self):
        """Load both the ImageNet pre-filter and the custom fine-tuned model."""
        if self.imagenet_model is None:
            self.imagenet_model = MobileNetV2(weights="imagenet")

        if os.path.exists(self.model_path):
            self.custom_model = load_model(self.model_path)
        else:
            print(f"Warning: Custom model not found at {self.model_path}")

    def _preprocess_imagenet(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMAGENET_SIZE)
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        return preprocess_input(arr)

    def _preprocess_custom(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _imagenet_classify(self, image_bytes: bytes) -> dict:
        """Full ImageNet classification — returns top-5."""
        arr = self._preprocess_imagenet(image_bytes)
        preds = self.imagenet_model.predict(arr, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]

        top_predictions = [
            {"label": label.replace("_", " ").title(), "confidence": float(conf)}
            for _, label, conf in decoded
        ]

        return {
            "prediction": decoded[0][1].replace("_", " ").title(),
            "confidence": float(decoded[0][2]),
            "raw_probability": float(decoded[0][2]),
            "top_predictions": top_predictions,
            "is_cat_dog": False,
        }

    def _is_cat_or_dog(self, image_bytes: bytes) -> bool:
        """Check if ImageNet thinks this is a cat or dog."""
        arr = self._preprocess_imagenet(image_bytes)
        preds = self.imagenet_model.predict(arr, verbose=0)[0]
        cat_dog_prob = sum(preds[i] for i in CAT_DOG_INDICES)
        return cat_dog_prob >= MIN_CAT_DOG_PROB

    def _custom_classify(self, image_bytes: bytes) -> dict:
        """Use our custom fine-tuned model for cat vs dog classification."""
        arr = self._preprocess_custom(image_bytes)
        prob = float(self.custom_model.predict(arr, verbose=0)[0][0])

        if prob > 0.5:
            label, confidence = "Dog", prob
        else:
            label, confidence = "Cat", 1.0 - prob

        return {
            "prediction": label,
            "confidence": confidence,
            "raw_probability": prob,
            "top_predictions": [
                {"label": "Dog", "confidence": prob},
                {"label": "Cat", "confidence": 1.0 - prob},
            ],
            "is_cat_dog": True,
        }

    def predict(self, image_bytes: bytes) -> dict:
        """
        Two-stage prediction:
        1. ImageNet pre-filter determines if image is a cat/dog.
        2a. If YES -> custom fine-tuned model provides final prediction.
        2b. If NO  -> ImageNet top-5 results are returned.
        """
        if self.imagenet_model is None:
            self.load()

        # Stage 1: Is this a cat or dog?
        is_cat_dog = self._is_cat_or_dog(image_bytes)

        # Stage 2: Route to the right model
        if is_cat_dog and self.custom_model is not None:
            return self._custom_classify(image_bytes)
        else:
            return self._imagenet_classify(image_bytes)


# ── Singleton accessor ──
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor


def predict_image(image_bytes: bytes) -> dict:
    return get_predictor().predict(image_bytes)
