# Intro to Deep Learning with Keras
Build, train, evaluate, and persist simple neural-network models with TensorFlow and Keras.

## Why this module matters
Many business datasets contain patterns that are difficult to capture with fixed rules or simple linear relationships. Deep learning provides a practical foundation for learning those patterns from examples. This module introduces the core model lifecycle in Keras: defining a network, configuring training, fitting data, measuring results, making the model available for reuse, and tracking experiments.

The examples use shallow multi-class classifiers so that the mechanics remain visible and testable:
- Build a model with the `Sequential` API.
- Build the same kind of model with Keras' Functional API.
- Compile a model with an optimizer, loss function, and metric.
- Train and evaluate the model.
- Save and load a complete model or only its weights.

## Business applications
With suitable data, validation, controls, and domain review, these techniques can support:
- flag unusual transactions for review, classify expenses or journal entries, and prioritize higher-risk samples for testing.
- support credit-risk segmentation, forecast-oriented classification, payment or claims triage, and detection of potentially fraudulent activity.
- segment clients or engagements, identify operational risk indicators, classify documents or requests, and prioritize cases for human analysis.
These are decision-support use cases. A model should inform professional judgment and established review procedures rather than replace auditors, finance professionals, or advisors.

## Limitations in real business settings
The exercises are intentionally small and do not by themselves address the requirements of a production or regulated system. Important limitations include:
- Results depend on representative, correctly labeled, sufficiently large, and legally usable data.
- Shallow examples may not represent the complexity, imbalance, drift, or time dependency of operational data.
- Accuracy alone is not enough for high-impact decisions; calibration, precision/recall, cost of errors, fairness, and explainability also matter.
- The examples do not provide data governance, privacy controls, security, model monitoring, versioning, deployment, or reproducible experiment tracking.
- Neural networks can learn historical bias and may be difficult to explain to clients, regulators, or audit committees.
- Predictions require human review, documented controls, clear ownership, and a process for handling uncertainty and model failure.

## Repository contents
| Script | Focus |
| --- | --- |
| `0-sequential.py` | Build a model with the Sequential API |
| `1-functional.py` | Build a model with the Functional API |
| `2-compile.py` | Configure the optimizer, loss, and metrics |
| `3-train.py` | Train a model with `fit` |
| `4-evaluate.py` | Evaluate a model with `evaluate` |
| `5-save_load_model.py` | Save and load the complete model |
| `6-save_load_weights.py` | Save and load model weights |
| `7-predict.py` | Generate predictions |
| `8-deep_nn_model.py` | Build a deeper neural network |
| `9-tensorboard.py` | Configure TensorBoard logging |

## Setup and workflow
### 1. Create and activate a Python environment
From this directory, create an isolated environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
On Windows, use `.venv\\Scripts\\activate` instead.

### 2. Install the required package (numpy, keras, matplotlib etc)
Keras is included with TensorFlow for these exercises:
```bash
python -m pip install --upgrade pip
python -m pip install tensorflow
```

Verify the installation:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 3. Execute the module workflow
Use a compatible dataset such as numeric features with one-hot encoded labels for the ten-class output used by the example models. The practical sequence is:
1. Import a model builder from `0-sequential.py` or `1-functional.py` and define the input size and hidden-layer width.
2. Compile the model with an appropriate optimizer, loss function, and evaluation metric in `2-compile.py`.
3. Train it on feature and label arrays with `3-train.py`.
4. Measure loss and accuracy on held-out data with `4-evaluate.py`.
5. Save and reload the full model with `5-save_load_model.py` when the architecture and optimizer state should travel together.
6. Recreate the architecture and save or reload only parameters with `6-save_load_weights.py` when the model definition is managed separately.
7. Use `7-predict.py` for inference and `9-tensorboard.py` to inspect training logs where applicable.

For a reliable assessment, keep training and test data separate, record the preprocessing applied to each dataset, and compare results against a simple baseline. Select the loss and metrics to match the business decision and the consequences of false positives and false negatives.

## Learning objectives
After completing this module, you should be able to explain:
- What Keras, TensorFlow, models, shallow networks, and deep networks are.
- When to use the Sequential API or the Functional API.
- What compiling, training, evaluating, and predicting do in Keras.
- How to choose an appropriate loss function, optimizer, and metric.
- How to save and restore a complete model or model weights.
- What TensorBoard is used for during model development.

## References
- [Keras documentation](https://keras.io/)
- [Keras 3 API documentation](https://keras.io/api/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [MIT Introduction to Deep Learning](https://introtodeeplearning.com/)
