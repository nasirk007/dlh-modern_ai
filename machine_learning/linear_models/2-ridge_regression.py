#!/usr/bin/env python3
"""module that evaluate regression model
using Scikit-learn.
"""
import numpy as np
from sklearn import linear_model


def ridge_regression(random_state):
    """to tune regression model using ridge regression.
    ridge regression is a regularization technique that adds
    a penalty term to the linear regression loss/cost function,
    which helps to prevent overfitting and improve
    the model's generalization performance.
    Ridge regression is particularly useful for overfitted problems
    or when coefficients are large in magnitude, and
    computation for model become complex. to reduce this complexity,
    ridge regression adds a penalty term to the cost/loss function.
    this loss or cost function is simply the error and more precisely
    its either residual sum of squares (RSS) (from adj R2) or
    mean squared error (MSE).
    """
    model = linear_model.Ridge(random_state=random_state)
    return model
