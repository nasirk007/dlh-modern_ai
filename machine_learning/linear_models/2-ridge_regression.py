#!/usr/bin/env python3
"""module that evaluate regression model
using Scikit-learn.
"""
import numpy as np
from sklearn import linear_model


def ridge_regression(random_state):
    """to tune regression model using ridge regression"""
    model = linear_model.Ridge(random_state=random_state)
    return model
