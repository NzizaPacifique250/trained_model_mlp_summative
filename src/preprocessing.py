"""
Image preprocessing and data generator utilities.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import io

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


def get_data_generators(train_dir, validation_dir, test_dir=None):
    """
    Creates data generators for training, validation, and optionally test.
    Training generator includes data augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    validation_generator = validation_test_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = validation_test_datagen.flow_from_directory(
            test_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=False,
        )

    return train_generator, validation_generator, test_generator


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess a single image for prediction.
    Handles JPEG, PNG, WebP, BMP, and other PIL-supported formats.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (handles RGBA, P, L, WebP, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
