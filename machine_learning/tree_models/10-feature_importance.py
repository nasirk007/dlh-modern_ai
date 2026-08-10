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
    # when we print importannces, we will see list of floats
    # for each feature/column but argsort is nuumpy function, will
    # returns the positions (indices) that would sort the importance values.
    # index == feature name == importance values
    # argsort function will look into importance values, then will
    # sort them from lowest to highest. In this exercise we have 13 features
    # and correspinding 13 importance values. let say at index 7 we have 0.0057
    # a lowest value, when sorted it will be at index 0, so argsort will return
    # 7 at index 0. and this is output of argsort function in total.
    indices = np.argsort(importances)
    return importances, indices
