#!/usr/bin/env python3
"""Computer Vision Applications.
This module provides tools and functions for object detection, image
segmentation, and dataset preprocessing using modern deep learning frameworks.
"""

import numpy as np
import albumentations as A


def basic_aug(image, bboxes, labels):
    """Applies basic Albumentations augmentations to an image
    and its bounding boxes. Performs seeded horizontal flips,
    brightness adjustments, and spatial affine transforms."""
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(translate_percent=0.1, scale=0.1, rotate=(-30, 0), p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        seed=42,
    )

    result = transform(image=image, bboxes=bboxes, labels=labels)

    # Albumentations returns lists and float labels, so convert back.
    return (
        result["image"],
        np.array(result["bboxes"]),
        [int(label) for label in result["labels"]],
    )
