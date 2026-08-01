#!/usr/bin/env python3
"""module that evaluate regression model
using Scikit-learn.
"""
from sklearn import linear_model


def lasso_regression(random_state):
    """to tune regression model using lasso regression."""
    model = linear_model.Lasso(random_state=random_state)
    return model
