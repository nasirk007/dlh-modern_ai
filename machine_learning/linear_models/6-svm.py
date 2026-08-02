#!/usr/bin/env python3
"""module that use shap library to explain the
predictions of a machine learning model."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """to tune regression model using support vector machine."""
    if not in ['linear', 'poly', 'rbf']:
        raise ValueError("Invalid kernel name")
    svm_model = svm.SVC(kernel=name, random_state=random_state)
    return svm_model
