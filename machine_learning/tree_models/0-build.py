#!/usr/bin/env python3
"""Module for decision tree model building."""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Build a decision tree model using the training data.

    Parameters: integer
    min_samples_leaf: mini no of samples required to be at a leaf node.
    min_samples_split: mini no of samples required to split an internal node.
    random_state: The seed used by the random number generator.
    """
    # Create a decision tree classifier
    clf = tree.DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return clf
