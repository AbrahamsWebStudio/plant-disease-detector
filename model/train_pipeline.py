"""Training pipeline for the plant disease detector project.

This module trains a simple image classifier using transfer learning with
MobileNetV2. It expects the PlantVillage-style dataset to be stored in the
project's dataset folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_OUTPUT_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODEL_OUTPUT_DIR / "class_names.json"
HISTORY_PATH = MODEL_OUTPUT_DIR / "training_history.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 5
SEED = 42


def validate_dataset(data_dir: Path) -> None:
    """Ensure the dataset folder exists and contains image subfolders."""
    if not data_dir.exists():
        raise FileNotFoundError(
            "Dataset folder not found. Please download the PlantVillage dataset "
            f"and place it in {data_dir}."
        )

    class_folders = [path for path in data_dir.iterdir() if path.is_dir()]
    if not class_folders:
        raise FileNotFoundError(
            f"No class folders were found in {data_dir}."
        )


def prepare_dataset(data_dir: Path, batch_size: int = BATCH_SIZE) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, tf.keras.preprocessing.image.DirectoryIterator, list[str]]:
    """Load and split the dataset into training and validation sets."""
    validate_dataset(data_dir)

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> keras.Model:
    """Create a transfer-learning model using MobileNetV2."""
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, class_names: list[str]) -> keras.callbacks.History:
    """Train the model and save the weights and metadata."""
    num_classes = len(class_names)
    model = build_model(num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)

    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(class_names, file_handle, indent=2)

    history_dict = history.history
    with HISTORY_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(history_dict, file_handle, indent=2)

    return history


def main() -> None:
    """Run the full training process."""
    train_ds, val_ds, class_names = prepare_dataset(DATASET_ROOT)
    train_model(train_ds, val_ds, class_names)
    print("Training finished successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
