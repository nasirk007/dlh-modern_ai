#!/usr/bin/env python3
"""
This module applies YOLO-compatible data augmentation
using Albumentations.
"""
import albumentations
import numpy as np


def basic_aug(image, bboxes, labels):
    """
    Apply YOLO-compatible data augmentation using Albumentations.
    The function must apply the following transformations
    - random horizontal flipping (p = 0.5)
    - brightness/contrast augmentation (p = 0.2)
    - Affine (translate_percent: 0.1, scale 0.1, rotate [-30, 0] with p = 0.5)

    Args:
        image (np.ndarray): Input image
        bboxes (List[List[int]]): Bounding boxes in Pascal VOC format
        labels (List[int]): Class labels corresponding to each bounding box

    Returns:
        the augmented image np.ndarray,
        augmented bounding boxes np.ndarray and labels List[int]
    """
    # Create the augmentation pipeline
    transform = albumentations.Compose(
        [
            # Flip the image horizontally with 50% probability
            albumentations.HorizontalFlip(p=0.5),
            # Randomly change brightness and contrast
            albumentations.RandomBrightnessContrast(p=0.2),
            # Move, resize, and rotate the image
            albumentations.Affine(
                translate_percent=0.1,
                scale=0.1,
                rotate=(-30, 0),
                p=0.5
            )
        ],
        # Tell Albumentations how the bounding boxes are formatted
        bbox_params=albumentations.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        ),
        # Make the random augmentation reproducible
        seed=42
    )

    # Apply augmentation to the image, boxes, and labels
    augmented = transform(
        image=image,
        bboxes=bboxes,
        labels=labels
    )

    # Get the augmented results
    augmented_image = np.array(augmented["image"])
    augmented_bboxes = np.array(augmented["bboxes"])
    augmented_labels = augmented["labels"]

    return augmented_image, augmented_bboxes, augmented_labels
