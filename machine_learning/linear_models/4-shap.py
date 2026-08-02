#!/usr/bin/env python3
"""module that use shap library to explain the
predictions of a machine learning model."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """X_train is used to initialize the explainer,
    while X_test is used to compute the SHAP values
    for the predictions made by the model. The function
    returns both the explainer object and the computed SHAP
    values for further analysis or visualization.

    Arguments:
    model: A trained regression model
    X_train: Input data used to initialize the explainer
    X_test: Input data to explain
    
    Returns:
    explainer: SHAP explainer object
    shap_values: SHAP values for the predictions on X_test
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
