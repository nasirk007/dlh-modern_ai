#!/usr/bin/env python3
"""
This module performs hyperparameter tuning to discover optimal training
hyperparameters, then trains and saves a final object detection model.
"""
import shutil
import yaml
from ultralytics import YOLO


def tune_hyperparameters():
    """
    Perform hyperparameter tuning,
    then trains and saves a final object detection model.

    Performs two-phase hyperparameter tuning and training:
    Phase 1: Lightweight hyperparameter search
    Phase 2: Continue training from best checkpoint

    Returns:
        None
    """
    # Dataset configuration file
    data_yaml = "datasets/detection/data.yaml"

    # Load the pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Define the hyperparameters that YOLO can explore
    search_space = {
        "lr0": (0.00001, 0.01),
        "lrf": (0.01, 1.0),
        "momentum": (0.7, 0.98),
        "weight_decay": (0.0, 0.001),

        "box": (1.0, 20.0),
        "cls": (0.1, 4.0),
        "dfl": (0.4, 12.0),

        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.9),

        "degrees": (0.0, 45.0),
        "translate": (0.0, 0.9),
        "scale": (0.0, 0.95),

        "mosaic": (0.0, 1.0),
        "mixup": (0.0, 1.0)
    }

    # Phase 1: Hyperparameter tuning
    model.tune(
        data=data_yaml,
        epochs=10,
        iterations=15,
        space=search_space,
        imgsz=640,
        batch=16,
        plots=False,
        val=True
    )

    # Files created automatically by YOLO tuning
    best_hyp_path = "runs/detect/tune/best_hyperparameters.yaml"
    best_checkpoint = "runs/detect/tune/weights/best.pt"

    # Read the best hyperparameters found during tuning
    with open(best_hyp_path, "r") as file:
        best_hyp = yaml.safe_load(file)

    # Phase 2: Final training
    # Load the best checkpoint from tuning
    best_model = YOLO(best_checkpoint)

    # Continue training using the best hyperparameters
    results = best_model.train(
        data=data_yaml,
        epochs=140,
        imgsz=640,
        batch=16,
        patience=20,
        **best_hyp
    )

    # Copy the best final model to the project folder
    shutil.copy(
        results.save_dir / "weights" / "best.pt",
        "best_model.pt"
    )

    return None
