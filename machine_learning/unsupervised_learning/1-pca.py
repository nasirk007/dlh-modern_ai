#!/usr/bin/env python3
"""
This module performs Principal Component Analysis on tabular data.
"""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
    Performs Principal Component Analysis (PCA) Bon tabular datan.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
        n_components (int, float or None):
            - int: Number of principal components to keep.
            - float (between 0 and 1): Minimum fraction of
              variance to preserve.
            - None: Keep all components.
        random_state (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Data transformed into principal component space.
        sklearn.decomposition.PCA: Fitted PCA instance.
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state)

    return pca.fit_transform(X), pca
