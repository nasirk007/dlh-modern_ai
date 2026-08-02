#!/usr/bin/env python3
"""module that use shap library to explain the
predictions of a machine learning model."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """to tune regression model using support vector machine."""
    model = svm.SVC(random_state=random_state)
    return model
