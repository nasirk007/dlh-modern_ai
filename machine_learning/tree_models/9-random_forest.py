#!/usr/bin/env python3
"""
This module creates a random forest classifier using Scikit-learn.
"""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """
    Create a random forest classifier using Scikit-learn.

    Args:
        n_estimators: Number of trees in the forest.
        random_state: Seed used by the random number
        generator for reproducibility.

    Returns:
        model: A Scikit-learn RandomForestClassifier instance.
    """
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
        )
    return model
