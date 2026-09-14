#!/usr/bin/env python3
"""Test runner for 4-transfer_101.py"""
import os
from tensorflow import keras

train_transfer_model = __import__('4-transfer_101').train_transfer_model

if __name__ == '__main__':
    # 1. Run the training function
    print("Starting training...")
    train_transfer_model()

    # 2. Check that the output file exists
    model_path = 'caltech101_model.h5'
    assert os.path.exists(model_path), f"Error: {model_path} not found!"
    print(f"Model successfully saved at: {model_path}")

    # 3. Load the saved model and evaluate on validation set
    model = keras.models.load_model(model_path)
    val_ds = keras.utils.image_dataset_from_directory(
        directory='101_ObjectCategories',
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=(224, 224),
        batch_size=32
    )

    loss, acc = model.evaluate(val_ds)
    print(f"Final Validation Loss: {loss:.4f}")
    print(f"Final Validation Accuracy: {acc * 100:.2f}%")

    if acc >= 0.85:
        print("Success: Target accuracy (>= 85%) achieved.")
    else:
        print(f"Warning: Accuracy {acc * 100:.2f}% is below 85% requirement.")
