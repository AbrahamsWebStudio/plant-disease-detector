"""Prediction utilities for the plant disease detector project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "plant_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"
IMAGE_SIZE = (224, 224)


def load_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return tf.keras.models.load_model(model_path)


def load_class_names(class_names_path: Path = CLASS_NAMES_PATH) -> List[str]:
    """Load class names from a JSON file."""
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    with class_names_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load and preprocess an image for model prediction."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image_path: Path) -> Tuple[str, float]:
    """Predict the most likely class for an image."""
    model = load_model()
    class_names = load_class_names()
    image_batch = preprocess_image(image_path)

    prediction = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence
