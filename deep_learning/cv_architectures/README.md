# Computer Vision Architectures

Practical PyTorch implementations of foundational CNN blocks and efficient image-classification architectures, from convolutional layers to MobileNet-style networks.

## Why This Module Matters

Convolutional neural networks (CNNs) are a foundation for extracting spatial patterns from images. Understanding their building blocks makes it possible to choose an architecture based on accuracy, latency, memory, and deployment constraints rather than treating a pretrained model as a black box.

This module focuses on the architecture decisions behind:

- Convolution, padding, stride, pooling, and activation layers
- Deeper networks with bottleneck and residual connections
- Efficient networks built with depthwise separable convolutions
- Training a CNN and connecting architecture choices to practical model behaviour

## Scope and Practical Limitations

The implementations are learning-focused architecture exercises. In a real business setting, additional work is required before using a computer vision model for decisions or automated controls:

- Performance depends on representative, correctly labelled image data and careful validation.
- Models can fail under changes in lighting, camera position, image quality, geography, or operating procedures.
- Deep CNNs may be difficult to explain to non-technical stakeholders and can encode bias from historical data.
- Training and inference can require significant compute, storage, monitoring, and specialist support.
- Accuracy alone is insufficient for high-impact decisions; thresholds, false positives, false negatives, and escalation paths must be governed.
- Sensitive images require appropriate consent, access control, retention, privacy, and security practices.
- Production use requires model versioning, drift monitoring, reproducible pipelines, human review, and periodic revalidation.

## Business Applications

With suitable domain data, computer vision architectures can support workflows in audit, finance, risk management, and advisory services, including:

- Reviewing document, receipt, invoice, or asset images for classification and prioritisation
- Supporting audit sampling and identifying potentially unusual visual records for human review
- Assessing images related to inventory, property, equipment, or collateral inspections
- Organising financial and operational documents before extraction or downstream analysis
- Supporting due diligence, portfolio monitoring, and business valuation evidence collection
- Classifying visual evidence used in risk assessments, compliance workflows, or advisory analysis

These are decision-support use cases. A model output should be treated as evidence or a review signal, not as a substitute for professional judgement, documented controls, or independent verification.

## Architectures Covered

- **Core CNN:** learns hierarchical visual features through convolution, activation, and pooling layers.
- **Bottleneck block:** uses $1\\times1$ convolutions to reduce and restore channel dimensions around a computationally expensive convolution.
- **Residual block:** uses skip connections to improve gradient flow and make deeper networks easier to optimise.
- **ResNet-101:** demonstrates a deep residual architecture built from bottleneck blocks.
- **Depthwise separable convolution:** separates spatial filtering from channel mixing to reduce parameters and computation.
- **MobileNet backbone:** provides a lightweight feature extractor for resource-constrained inference.
- **MobileNetV1:** applies depthwise separable convolutions to build an efficient image-classification network.

## Project Workflow

1. Create and activate an isolated Python environment.
2. Install the required deep learning and numerical computing packages Numpy, tensorflow, and matplotlit
3. Inspect each task's expected class, method, tensor shape, and constructor arguments.
4. Implement the architecture using PyTorch modules and preserve the required public interfaces.
5. Validate tensor dimensions with small synthetic inputs before training.
6. Train the CNN using a reproducible data split, loss function, optimiser, and evaluation loop.
7. Compare architecture complexity using parameter count, computational cost, accuracy, and inference speed.
8. Record limitations and validation results before considering a domain-specific application.

## Task Overview

```text
cv_architectures/
├── 0-create_convolutional_architecture.py  # Build a basic convolutional architecture
├── 1-train_cnn.py                          # Train and evaluate a CNN
├── 2-bottleneck_block.py                   # Implement a bottleneck residual block
├── 3-resnet_101.py                         # Build a ResNet-101-style network
├── 4-depthwise_separable_conv_block.py     # Implement an efficient convolution block
├── 5-mobilenet_backbone.py                 # Build a lightweight MobileNet backbone
├── 6-mobilenet_v1.py                       # Assemble a MobileNetV1-style classifier
└── README.md
```

The numbered filenames above describe the intended task organisation. Keep the exact filenames, class names, method signatures, and output contracts required by the project checker when adding the implementations.

## Setup

The commands below assume Python 3.11 and a Unix-like shell. Run them from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow
```

If the environment provides a CUDA-enabled PyTorch build, install the matching packages from the official PyTorch selector instead of replacing them with an incompatible wheel. CPU execution is sufficient for shape checks and small experiments.

## Suggested Validation Checklist

- Confirm every Python file starts with the required shebang, ends with a newline, and is executable.
- Run syntax and style checks before submission.
- Test each model with a small input tensor and verify its output shape.
- Confirm training loss and evaluation metrics are computed on separate data.
- Check parameter counts and inference time for each architecture.
- Test behaviour with invalid tensor shapes and document expected input conventions.

## Skills Developed

- Translate CNN theory into reusable PyTorch modules
- Reason about tensor shapes, receptive fields, channels, strides, and padding
- Build bottleneck and residual blocks for deep networks
- Reduce model cost with depthwise and pointwise convolutions
- Train and evaluate image-classification models reproducibly
- Compare architecture trade-offs across accuracy, parameters, FLOPs, memory, and latency
- Identify governance, privacy, robustness, and explainability requirements for business deployment

## Learning References

- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Dive into Deep Learning: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
