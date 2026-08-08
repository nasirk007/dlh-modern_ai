#!/usr/bin/env python3
"""Module to do prediction with a decision tree model."""


def generate_predictions(clf, X):
    """Make predictions using a trained decision tree classifier.
    Parameters:
    clf: A trained Scikit-learn classifier instance
    X: Feature matrix (NumPy array or pandas DataFrame)
    Return: A NumPy array containing the predicted class
    labels for the input samples.
    """
    predictions = clf.predict(X)
    return predictions
