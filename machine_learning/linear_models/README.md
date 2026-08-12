# Linear Models Module

Concise summary: A compact collection of linear regression and classification examples (Ordinary Least Squares, Ridge, Lasso, Logistic) with explainability via SHAP for model interpretation.

## Overview
- Purpose: teach core linear modeling techniques for regression and binary classification, evaluation metrics, regularization, and model explainability using SHAP.

## Why use linear models?
- Linear models are fast, interpretable, and often effective for well-conditioned tabular data. They provide baseline performance, coefficient-based insights, and are a strong first-step for finance and audit workflows.

## How linear models differ from other ML types
- Compared with tree-based or deep models, linear methods assume (or approximate) a linear relationship between inputs and outputs, producing coefficients that are directly interpretable. They are typically faster, less prone to overfitting on small datasets (with proper regularization), and easier to explain.

## Algorithms and their role
- `LinearRegression` — Ordinary Least Squares regression; baseline predictive model for continuous targets.
- `Ridge` — L2-regularized regression; stabilizes coefficients and reduces variance.
- `Lasso` — L1-regularized regression; encourages sparse coefficients for feature selection.
- `LogisticRegression` — linear classifier for binary targets; outputs probabilities and decision boundaries.
- `SVM (linear/kernel)` — linear or kernelized classifiers for tougher decision boundaries (mentioned as reference).
- `SHAP` — model-agnostic explainability; quantifies feature contributions to individual predictions.

## Task functions (names, arguments, returns)
- `Linear_Regression()`
	- Args: None
	- Returns: `model` — untrained `sklearn.linear_model.LinearRegression` instance
	- Imports: `from sklearn import linear_model`

- `evaluation_metrics_for_regression(y_true, y_pred)`
	- Args: `y_true` (1D numpy array), `y_pred` (1D numpy array)
	- Returns: tuple `(mse, rmse, mae, r2)` using `sklearn.metrics` and `numpy`
	- Imports: `from sklearn import metrics`, `import numpy as np`

- `ridge_regression(random_state)`
	- Args: `random_state` (int)
	- Returns: `model` — untrained `sklearn.linear_model.Ridge` instance
	- Imports: `from sklearn import linear_model`

- `lasso_regression(random_state)`
	- Args: `random_state` (int)
	- Returns: `model` — untrained `sklearn.linear_model.Lasso` instance
	- Imports: `from sklearn import linear_model`

- `get_shap_explainer_and_values(model, X_train, X_test)`
	- Args: `model` (trained model), `X_train` (background data), `X_test` (data to explain)
	- Returns: tuple `(explainer, shap_values)` where `explainer` is a SHAP explainer and `shap_values` are the explanations for `X_test`
	- Imports: `import shap`

- `Logistic_Regression_Model(random_state)`
	- Args: `random_state` (int)
	- Returns: `model` — untrained `sklearn.linear_model.LogisticRegression` instance
	- Imports: `from sklearn import linear_model`

## Finance / Audit / Investment Applications (high-level)
- Credit risk scoring and probability-of-default estimation (classification/regression hybrid)
- Loan and portfolio valuation modeling using linear regressors for explainable forecasts
- Expense and revenue trend modelling for auditing and anomaly detection
- Feature-driven valuation drivers in business valuation and sensitivity analysis
- Simplified factor models for investment signals and risk factor decomposition

## Project workflow
1. Inspect and visualize the dataset (`visualize_data.py`, `explore_features_target_correlation.py`).
2. Split data into train / test (and optionally validation).
3. Create baseline `LinearRegression()` model and evaluate with `evaluation_metrics_for_regression`.
4. Train regularized models (`ridge_regression`, `lasso_regression`) and compare metrics.
5. For classification tasks, build `Logistic_Regression_Model` and evaluate classification metrics.
6. Fit a final model and run `get_shap_explainer_and_values` for feature-level explanations.
7. Summarize results, coefficients, and SHAP visualizations for stakeholder reports.

## Task overview (tree)
```
linear_models/
├── 0-linear_regression.py      # Linear_Regression(): return LinearRegression() instance
├── 1-evaluation_metrics.py     # evaluation_metrics_for_regression(y_true, y_pred)
├── 2-ridge.py                  # ridge_regression(random_state)
├── 3-lasso.py                  # lasso_regression(random_state)
├── 4-shap_explain.py           # get_shap_explainer_and_values(model, X_train, X_test)
├── 5-logistic.py               # Logistic_Regression_Model(random_state)
├── visualize_data.py           # visualization helper (3D scatter)
├── explore_features_target_correlation.py # scatter plots for feature vs target
└── README.md
```

## Dataset & visualization
- This module includes small example scripts to visualize feature relationships and 3D scatter plots. Use `visualize_data.py` and `explore_features_target_correlation.py` (included in the folder) to inspect synthetic or local datasets before modeling.

## Requirements
- Python 3.11 (Ubuntu 20.04 LTS execution assumptions)
- Required packages (suggested pinned versions):
	- `numpy==2.0.2`
	- `scikit-learn==1.6.1`
	- `shap==0.48.0`
	- `pillow==11.3.0`

## Skills developed
- Implement and evaluate linear regression and classification models
- Apply L1/L2 regularization (Lasso and Ridge) and understand trade-offs
- Compare models using regression and classification metrics (MSE, RMSE, MAE, R2, precision/recall/AUC)
- Use SHAP to explain feature contributions and produce stakeholder-ready visualizations
- Prepare reproducible, documented Python modules and baseline modeling workflows

## Quick start
1. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy==2.0.2 scikit-learn==1.6.1 shap==0.48.0 pillow==11.3.0
```

2. Visualize data:

```bash
./visualize_data.py
./explore_features_target_correlation.py
```

3. Run example scripts (each task file contains a main or callable function to run experiments).

---
Concise, public-facing documentation for linear modelling with explainability and finance-oriented use cases.