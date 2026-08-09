#!/usr/bin/env python3
"""Retrieve the Pruning Path of a Decision Tree."""


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

    alpha is simply a penalty to complexity of tree.
    ccp_alphas is a hyperparameter that controls the complexity of
    the model after being built this task is analyse the trade-off
    btw ccp_alpha and the accuracy of the model.
    The higher the ccp_alpha, the model pruned more which can lead to
    a simpler model with less overfitting. However, if ccp_alpha is
    too high, the model may become too simple and underfit.
    larger alpha penalises the complexity of the model more,
    leading to a simpler model. In the model each node has alpha
    value and node with the lowest alpha value is pruned first.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    return ccp_alphas, impurities
