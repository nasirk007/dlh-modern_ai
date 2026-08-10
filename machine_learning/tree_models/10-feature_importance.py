#!/usr/bin/env python3
"""
This module computes and returns the feature importances from
a trained random forest model.
"""
import numpy as np


def feature_importance(rf):
    """
    Returns the feature importances from a trained random forest model.

    Args:
        rf: A trained Scikit-learn RandomForestClassifier instance.

    Returns:
        importances: A NumPy array of feature importance scores.
        indices: A NumPy array of feature indices sorted from least
        to most important (ascending order).
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
