#!/usr/bin/env python3
"""
This module initializes and returns an untrained boosting classifier
based on the specified algorithm name.
"""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Returns an untrained boosting classifier
    based on the specified algorithm name.
    - Initialize the selected model with the
    specified n_estimators and random_state.
    - For LightGBM, verbose is set to -1 to suppress training logs.
    - Raise a ValueError with the message “Unknown model name '{name}'”
    if the provided model name is invalid.

    Args:
        name (str): Name of the boosting algorithm. Must be one of:
        - 'adaboost' — returns an AdaBoostClassifier.
        - 'gradientboosting' — returns a GradientBoostingClassifier.
        - 'xgboost' — returns an XGBClassifier
        - 'lightgbm' — returns an LGBMClassifier
        n_estimators (int): Number of boosting iterations (trees).
        random_state (int): Random seed for reproducibility.

    Returns:
        An untrained instance of the selected boosting classifier.
    """
    if name == "adaboost":
        return ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state)
    elif name == "gradientboosting":
        return ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state)
    elif name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state)
    elif name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1)
    else:
        raise ValueError(f"Unknown model name '{name}'")
