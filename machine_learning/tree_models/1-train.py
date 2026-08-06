#!/usr/bin/env python3
"""Module for training decision tree model."""


def train_tree(clf, X, y):
    """Build a decision tree model using the training data.
    Parameters:
    clf: A Scikit-learn classifier instance
    X: Input features
    y: Target labels
    Return: None
    """
    # train the decision tree classifier
    clf.fit(X, y)
    return None
