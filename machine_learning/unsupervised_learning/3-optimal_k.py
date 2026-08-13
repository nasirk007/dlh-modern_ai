#!/usr/bin/env python3
"""
This module evaluates K-Means clustering quality to measure
cluster cohesion and separation, and to compute the inertia
to be used for the elbow method.
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering quality using silhouette scores
    to measure cluster cohesion and separation, and to compute
    the inertia to be used for the elbow method.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        max_clusters (int): Maximum number of clusters to evaluate (>=2).
        random_state (int): Random seed for reproducibility.

    Returns:
        list[int]: Evaluated cluster numbers.
        list[float]: Inertia values corresponding to each cluster number
                     for the elbow method.
        list[float]: Silhouette scores corresponding to each cluster number
                     for cluster quality evaluation.
    """
    k_values = []
    inertias_values = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        model = K_Means(X, k, random_state)

        k_values.append(k)
        inertias_values.append(model.inertia_)
        silhouette_scores.append(metrics.silhouette_score(
            X, model.labels_))

    return k_values, inertias_values, silhouette_scores
