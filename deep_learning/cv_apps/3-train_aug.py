#!/usr/bin/env python3
"""Computer Vision Applications.
This module provides tools and functions for object detection, image
segmentation, and dataset preprocessing using modern deep learning frameworks.
"""
from ultralytics import YOLO


def train_with_augmentation(
        data, model_path="yolov8n.pt", epochs=50, imgsz=640, batch=16,
        augmentation=True, yolo_aug_params=None,
        albumentations_transforms=None, save=False, plots=False,
        verbose=False):
    """Trains a YOLO model using configurable native
    or custom Albumentations augmentations. Executes model training with
    specified hyperparameters and returns the model and results."""
    yolo_model = YOLO(model_path)
    no_aug = {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.0, "bgr": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "auto_augment": None, "erasing": 0.0,
    }

    train_args = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }
    if not augmentation:
        train_args.update(no_aug)
    if yolo_aug_params:
        train_args.update(yolo_aug_params)
    if albumentations_transforms:
        train_args["augmentations"] = albumentations_transforms

    results = yolo_model.train(**train_args)
    return yolo_model, results
