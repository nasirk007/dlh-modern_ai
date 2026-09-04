# Transfer Learning with Keras

Adapt an ImageNet-pretrained MobileNetV2 to a new image-classification task with feature extraction, augmentation, and selective fine-tuning.

## Why this module matters

Training a deep vision model from scratch can require substantial labelled data, compute, and time. Transfer learning reuses visual features learned from a large source dataset and adapts them to a smaller target dataset. This module demonstrates the practical progression from a frozen backbone to a fine-tuned classifier while keeping the number of trainable parameters controlled.

## Business applications

With representative data and appropriate review controls, transfer learning can support:

- Classifying invoices, receipts, forms, and other business documents.
- Prioritizing visual records for audit, compliance, or quality review.
- Categorizing inventory, equipment, property, or collateral images.
- Supporting due diligence, inspection, and operational triage workflows.
- Organizing visual evidence before downstream extraction or advisory analysis.

These are decision-support applications. Predictions should be reviewed against documented thresholds and professional judgement, especially when errors have financial, regulatory, or customer impact.

## Limitations in real business settings

Transfer learning is not a substitute for representative data or production controls:

- ImageNet features may not transfer well when the source and target domains are substantially different.
- Small or biased target datasets can produce unstable results and reinforce historical bias.
- Freezing too many layers can limit adaptation; unfreezing too many can overfit or damage useful pretrained features.
- Accuracy may hide important class-specific errors, false positives, false negatives, and calibration problems.
- Images can contain sensitive information requiring consent, access controls, retention policies, and secure processing.
- Real deployment also requires independent testing, drift monitoring, model versioning, reproducible preprocessing, latency checks, and human escalation paths.

## Tasks and workflow

| Script | Focus |
| --- | --- |
| `0-frozen_extractor.py` | Load ImageNet MobileNetV2 without its classifier and freeze the backbone |
| `1-classification_head.py` | Add a trainable classification head to extracted features |
| `2-unfreeze_top.py` | Unfreeze only the top layers for controlled fine-tuning |
| `3-data_aug.py` | Build a seeded image-augmentation pipeline |
| `4-transfer_101.py` | Train and fine-tune a Caltech-101 image classifier |

The main training script uses 224x224 RGB images, a 20% validation split, a frozen MobileNetV2 stage, and a lower-learning-rate fine-tuning stage. It saves the resulting model as `caltech101_model.h5`.

## Setup

Run the commands below from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorflow pillow
```

Place the Caltech-101 image folders under `101_ObjectCategories/` when running `4-transfer_101.py`. The script expects one folder per class and downloads pretrained MobileNetV2 weights when they are not already available locally. Confirm that the Python and TensorFlow versions are compatible with your operating system before installation.

## Recommended sequence

1. Build the frozen MobileNetV2 feature extractor with `0-frozen_extractor.py`.
2. Attach a task-specific dense classification head with `1-classification_head.py`.
3. Add the seeded flip, rotation, zoom, and contrast transformations from `3-data_aug.py`.
4. Train the new head while keeping the pretrained backbone frozen.
5. Unfreeze only selected top layers with `2-unfreeze_top.py` and recompile with a smaller learning rate.
6. Fine-tune on the validation-monitored target task using early stopping and learning-rate reduction.
7. Evaluate on held-out data and record class-level performance, preprocessing, trainable layers, and model version.

## Learning objectives

After completing this module, you should be able to explain:

- What transfer learning, feature extraction, and fine-tuning are.
- How to choose a pretrained model and assess source-target similarity.
- Why early CNN layers are generally reusable while later layers are more task-specific.
- How dataset size affects the decision to freeze or unfreeze layers.
- How feature reuse, data augmentation, and regularization can reduce overfitting.
- Why fine-tuning should normally use only part of the network and a smaller learning rate.

## References

- [Keras transfer learning and fine-tuning guide](https://keras.io/guides/transfer_learning/)
- [Keras MobileNetV2 application](https://keras.io/api/applications/mobilenet/)
- [CS231n transfer learning notes](https://cs231n.github.io/transfer-learning/)
- [TensorFlow image classification](https://www.tensorflow.org/tutorials/images/classification)
