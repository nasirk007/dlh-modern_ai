# Enhancing Deep Learning Models
Improve neural-network training and generalization with optimization, regularization, and hyperparameter-tuning techniques in TensorFlow and Keras.

## Why this module matters
A model can have a suitable architecture and still perform poorly because it is trained with an unsuitable optimizer, learning rate, initialization strategy, or regularization approach. This module focuses on the practical controls that make training more stable and help reduce overfitting:
- Compare batch, stochastic, and mini-batch gradient descent.
- Configure momentum and Nesterov momentum with SGD.
- Compare SGD with adaptive optimizers such as Adam and RMSprop.
- Apply exponential and inverse-time learning-rate schedules.
- Select weight initializers for common activation functions.
- Reduce overfitting with L2 regularization, dropout, and early stopping.
- Define and search tunable model hyperparameters with KerasTuner.

## Business applications
These techniques can improve decision-support models used to:
Prioritize transactions or journal entries for testing, identify unusual activity, and classify expense or control exceptions.
Support credit-risk segmentation, fraud or payment triage, claims classification, and risk-prioritization workflows.
Classify client requests or documents, segment engagements, identify operational risk indicators, and prioritize cases for professional review.

Optimization and regularization do not create business value on their own. They can help a model learn more reliably from suitable data, but outputs should remain subject to domain expertise, documented controls, and human review.

## Limitations in real business settings
The exercises demonstrate model-training techniques, not a complete production or regulated machine-learning system. In real settings:
- Model quality depends on representative, sufficiently large, correctly labeled, and legally usable data.
- Validation results may not reflect changing populations, rare events, data drift, or the cost of different error types.
- Better training loss does not necessarily mean better business outcomes; accuracy, precision, recall, calibration, fairness, and stability should also be assessed.
- Hyperparameter searches can be computationally expensive and may overfit the validation data when the search process is not independently evaluated.
- Neural networks can reproduce historical bias and may be difficult to explain to clients, regulators, or audit committees.
- The examples do not implement data governance, privacy and security controls, deployment, monitoring, model versioning, or approval workflows.

For high-impact decisions, use independent test data, document assumptions and preprocessing, establish performance thresholds, monitor the model after release, and define escalation procedures for uncertain or incorrect predictions.

## Repository contents
`0-gradient_descent_variants.py` | Configure batch, stochastic, and mini-batch SGD |
| `1-momentum_sgd_variants.py` | Configure SGD with momentum and Nesterov momentum |
| `2-adaptive_optimizers.py` | Configure SGD, Adam, and RMSprop |
| `3-learning_rate_schedule.py` | Apply exponential and inverse-time schedules |
| `4-weight_initialization.py` | Select initializers for activation functions |
| `5-l2_reg.py` | Add L2 kernel regularization |
| `6-dropout.py` | Add input and hidden-layer dropout |
| `7-early_stopping.py` | Create an early-stopping callback |
| `8-build_model_to_be_tuned.py` | Build a tunable classification model |
| `9-initiate_tuner.py` | Initialize Hyperband, random-search, or Bayesian tuners |
| `10-search.py` | Search for the best hyperparameters |

## Setup and workflow
### 1. Create a Python environment
From this directory, create and activate an isolated environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
On Windows, activate the environment with `.venv\\Scripts\\activate`.

### 2. Install the dependencies
The scripts require TensorFlow and KerasTuner:
```bash
python -m pip install --upgrade pip
python -m pip install tensorflow keras-tuner
```

Verify both packages:
```bash
python -c "import tensorflow as tf; import keras_tuner; print(tf.__version__, keras_tuner.__version__)"
```

If TensorFlow installation fails, check that the selected Python version and operating system are supported by the TensorFlow release you are installing.

### 3. Complete the exercises
Use numeric feature data and labels compatible with the ten-class model outputs used in the examples. Keep training, validation, and test data separate where the exercise allows it.
1. Start with `0-gradient_descent_variants.py` and compare the batch size used by each SGD variant.
2. Use `1-momentum_sgd_variants.py` to compare plain SGD, momentum, and Nesterov momentum.
3. Use `2-adaptive_optimizers.py` to configure SGD, Adam, or RMSprop, then compare their behavior on the same model and data.
4. Use `3-learning_rate_schedule.py` to apply exponential or inverse-time decay to SGD.
5. Use `4-weight_initialization.py` to pair Glorot initialization with sigmoid or tanh and He initialization with ReLU-style activations.
6. Use `5-l2_reg.py` and `6-dropout.py` to compare regularized models with an unregularized baseline.
7. Use `7-early_stopping.py` with validation metrics to stop training and restore the best observed weights.
8. Build a tunable model with `8-build_model_to_be_tuned.py`, create a tuner with `9-initiate_tuner.py`, and run the search from `10-search.py`.
9. Evaluate the selected configuration on held-out data and record the optimizer, schedule, initializer, regularization settings, and validation results.

## Learning objectives
After completing this module, you should be able to explain:
- What optimization is and how SGD, momentum, Nesterov momentum, and adaptive optimizers work.
- When Adam may be preferred over SGD and why optimizer choice should be tested empirically.
- What learning-rate schedules do and how they affect training.
- Why weights need initialization and how initialization relates to activation functions.
- What overfitting, L2 regularization, dropout, and early stopping are.
- What hyperparameter tuning is, how to build a tunable model, and how to choose among tuner types.
- How training loss differs from validation loss and why both should be monitored.

## References
- [Keras optimizers](https://keras.io/api/optimizers/)
- [Keras learning-rate schedules](https://keras.io/api/optimizers/learning_rate_schedules/)
- [Keras regularizers](https://keras.io/api/layers/regularizers/)
- [Keras callbacks](https://keras.io/api/callbacks/)
- [KerasTuner documentation](https://keras.io/keras_tuner/)
- [TensorFlow documentation](https://www.tensorflow.org/)
