#!/usr/bin/env python3
"""Retrieve the Pruning Path of a Decision Tree."""
from sklearn import model_selection


def get_pruning_path(clf, X, y):
    """Function that retrieves the cost-complexity pruning
    path for a given decision tree classifier.

    Arguments:
    clf: A DecisionTreeClassifier instance
    X: Input features
    y: Target labels

    Returns:
    ccp_alphas: A NumPy array containing the effective alpha
    values used for pruning (post prunning).
    impurities: A NumPy array containing the total impurity of
    leaves at each corresponding alpha value.
    """
    # retrieve the cost-complexity pruning path
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    return ccp_alphas, impurities
