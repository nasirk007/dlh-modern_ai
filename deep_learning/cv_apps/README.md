# Computer Vision Applications

Build practical object-detection and image-segmentation workflows with YOLO, OpenCV, Albumentations, and NumPy.

## Why this module matters

Computer vision systems can turn images and video into structured signals for review and analysis. This module focuses on the workflow around a vision model: organizing data, applying bounding-box-aware augmentation, training and tuning a detector, and adjusting inference behavior. It also introduces the concepts needed to interpret instance-segmentation outputs, including masks, IoU, non-max suppression, and segmentation mAP.

## Business applications

With representative data, appropriate controls, and human review, these techniques can support:

- Detecting and prioritizing unusual visual evidence in audit and compliance workflows.
- Classifying and locating objects in invoices, receipts, forms, equipment, inventory, and property images.
- Supporting inspections, quality checks, safety reviews, and operational exception triage.
- Segmenting assets or regions of interest for due diligence, risk assessment, and advisory analysis.
- Measuring visual process performance where image-based evidence is available.

Predictions should be treated as review signals or evidence, not as a replacement for professional judgement or independent verification.

## Limitations in real business settings

These exercises focus on the model-development workflow and do not constitute a production computer-vision system. Real deployments also require:

- Representative, correctly labelled data covering expected environments and edge cases.
- Robustness testing across lighting, camera position, image quality, geography, and operating conditions.
- Evaluation beyond a single score, including class-level precision and recall, false-positive cost, false-negative cost, IoU, and mAP.
- Privacy, consent, access control, retention, encryption, and secure handling of sensitive images.
- Versioned datasets, reproducible preprocessing, model monitoring, drift detection, latency checks, and escalation procedures.
- Review of bias, explainability, and regulatory obligations before using outputs in high-impact decisions.

## Task workflow

| Task | Focus |
| --- | --- |
| `0` | Download and organize the segmentation dataset |
| `1` | Apply basic image and bounding-box transformations |
| `2` | Build custom Albumentations transformations |
| `3` | Train a YOLO model with augmentation |
| `4` | Tune training hyperparameters |
| `5` | Tune inference settings and evaluate predictions |

The current augmentation implementation, `1-basic_aug.py`, applies horizontal flipping, brightness/contrast adjustment, and affine transformations while keeping Pascal VOC bounding boxes and labels aligned with the image.

## Setup

The project targets Ubuntu 20.04 LTS with Python 3.11. Run these commands from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
	numpy==2.0.2 \
	matplotlib==3.10.0 \
	opencv-python==4.12.0.88 \
	torch==2.8.0 \
	albumentations==2.0.8 \
	ultralytics==8.4.7
```

For GPU training, install the PyTorch build compatible with the available CUDA version using the official PyTorch installation instructions. CPU execution is suitable for small augmentation and inference checks.

Verify the main dependencies:

```bash
python -c "import albumentations, cv2, numpy, torch, ultralytics; print('Environment ready')"
```

## Dataset setup

Download the [segmentation dataset](https://intranet-dlh.hbtn.io/rltoken/7C3Z88hDdV1Wc-xZZDrf-w), extract it, and organize it under:

```text
datasets/segmentation/
```

Before training, confirm that images, annotations, class names, and train/validation splits match the dataset configuration expected by the YOLO workflow. Do not commit private or restricted datasets to the repository.

## Recommended workflow

1. Inspect the dataset structure, annotation format, class balance, and image dimensions.
2. Organize the segmentation data under `datasets/segmentation/` and validate that annotations map to the correct images.
3. Apply basic and custom Albumentations transforms while transforming bounding boxes or masks consistently with each image.
4. Train the YOLO model with a reproducible configuration and separate validation data.
5. Tune learning rate, batch size, image size, augmentation, and other relevant parameters using validation metrics.
6. Tune confidence and IoU thresholds during inference, then evaluate on held-out data.
7. Compare detection and segmentation results using precision, recall, IoU, and the appropriate bounding-box or mask mAP.
8. Record the dataset version, configuration, model version, metrics, and known failure cases.

## Learning objectives

After completing this module, you should be able to explain:

- What object detection and single-shot detection are, and how YOLO performs detection.
- How IoU, anchor boxes, non-max suppression, and mAP are used in detection evaluation.
- The difference between object detection, semantic segmentation, and instance segmentation.
- What segmentation masks and polygon annotations represent.
- The difference between bounding-box IoU/mAP and mask IoU/mAP.
- How non-max suppression is applied when detected masks overlap.
- When object detection is more appropriate than segmentation, and when segmentation is justified.

## Code requirements

All submitted Python files should:

- Start with exactly `#!/usr/bin/env python3` and end with a newline.
- Follow `pycodestyle` version 2.14.0.
- Include module, class, and function documentation where applicable.
- Be executable with `chmod +x filename.py`.
- Keep public function signatures and output formats required by the project checker.

Run a style check with:

```bash
pycodestyle *.py
```

## Selected references

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [Albumentations documentation](https://albumentations.ai/docs/)
- [OpenCV documentation](https://docs.opencv.org/)
- [IoU and object-detection metrics](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/)
- [Non-max suppression](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/)
