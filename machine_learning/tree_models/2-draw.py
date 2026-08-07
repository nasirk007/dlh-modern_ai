#!/usr/bin/env python3
"""Module to draw decision tree model."""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """Displays textual structure of a trained decision
    tree classifier using Scikit-learn..
    Parameters:
    clf: A trained decision tree classifier instance
    feature_names: List of feature names for all x columns
    class_names: List of class names in y column
    Print: Textual representation of the decision tree structure
    """
    tree_rules = tree.export_text(clf,
                                  feature_names=list(feature_names),
                                  class_names=list(class_names))
    print(tree_rules)
