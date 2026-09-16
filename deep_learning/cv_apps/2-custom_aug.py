#!/usr/bin/env python3
"""Computer Vision Applications.
This module provides tools and functions for object detection, image
segmentation, and dataset preprocessing using modern deep learning frameworks.
"""

import numpy as np
import albumentations as A


def custom_aug(image, bboxes, labels):
    """Applies custom Albumentations blur and spatial distortion transforms.
    Modifies an image and its bounding boxes using motion blur
    and elastic or optical effects."""
    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=5, p=0.9),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                    A.OpticalDistortion(distort_limit=0.05, p=0.2),
                ],
                p=0.9,
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        seed=42,
    )

    result = transform(image=image, bboxes=bboxes, labels=labels)

    # Albumentations returns lists and float labels, so convert back.
    return (
        result["image"],
        np.array(result["bboxes"]),
        list(result["labels"]))
