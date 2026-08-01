#!/usr/bin/env python3
"""creating logistic regression model
using Scikit-learn.
"""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """to tune regression model using logistic regression."""
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
