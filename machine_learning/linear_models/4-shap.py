#!/usr/bin/env python3
"""module that use shap library to explain the
predictions of a machine learning model."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """to get shap explainer and values for the
    machine learning model. It uses training model
    and test data to get the shap values.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
