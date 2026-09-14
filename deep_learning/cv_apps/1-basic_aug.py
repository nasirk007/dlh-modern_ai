#!/usr/bin/env python3
"""
This module applies YOLO-compatible data augmentation
using Albumentations.
"""
import albumentations as A
import numpy as np
import cv2


def basic_aug(image, bboxes, labels):
    """
    Apply YOLO-compatible data augmentation using Albumentations.
    """
    # Create the augmentation pipeline with exact parameters
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                translate_percent=0.1,
                scale=0.1,
                rotate=(-30, 0),
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=0,
                crop_border=False,
                keep_size=True
            )
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        ),
        seed=42
    )

    # Apply augmentation
    augmented = transform(
        image=image,
        bboxes=bboxes,
        labels=labels
    )

    return (
        np.array(augmented["image"]),
        np.array(augmented["bboxes"]),
        augmented["labels"]
    )
