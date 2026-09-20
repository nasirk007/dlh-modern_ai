#!/usr/bin/env python3
"""Computer Vision Applications.
This module provides tools and functions for object detection, image
segmentation, and dataset preprocessing using modern deep learning frameworks.
"""
from ultralytics import YOLO


def inference_tuning(
        data_yaml, model,
        conf_list=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        iou_list=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        imgsz=640):
    """Test every conf and iou pair and collect the scores.
    model can be a path to weights or an already loaded YOLO model.
    Returns a list of dicts, one per combination tried."""
    if isinstance(model, str):
        model = YOLO(model)

    results = []
    for conf in conf_list:
        for iou in iou_list:
            metrics = model.val(
                data=data_yaml, conf=conf, iou=iou, imgsz=imgsz,
                plots=False, verbose=False)
            results.append({
                "conf": conf,
                "iou": iou,
                "map50": metrics.box.map50,
                "map50_95": metrics.box.map,
            })
    return results
