#!/usr/bin/env python3
"""Module to draw decision tree model."""
import matplotlib.pyplot as plt
from sklearn.tree import export_text


def draw(clf, feature_names, class_names):
    """Displays textual structure of a trained decision
    tree classifier using Scikit-learn..
    Parameters:
    clf: A trained decision tree classifier instance
    feature_names: List of feature names
    class_names: List of class names
    Return: None
    Print: Textual representation of the decision tree structure
    """
    tree_rules = export_text(clf, feature_names=feature_names)
    print(tree_rules)
