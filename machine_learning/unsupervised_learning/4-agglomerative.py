"""4. Agglomerative Hierarchical Clustering
Write a function Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True) that performs Agglomerative hierarchical clustering on tabular data using Scikit-learn.

The function will perform three main tasks:

Dimensionality reduction (optional): Apply PCA to reduce the data to n_components dimensions if use_pca_data=True.
Clustering: Fit an Agglomerative Clustering model on the (PCA-reduced or original) data using Ward linkage.
Evaluation: Compute the silhouette score for the clustering (if n_clusters > 1).
Arguments:

X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
n_clusters (int): Number of clusters
random_state (int): Random seed for reproducibility
n_components (int): Number of PCA components to reduce the data to (used only if use_pca_data=True)
use_pca_data (bool, default=True): Whether to apply PCA to reduce dimensionality before clustering
Returns:

sklearn.cluster.AgglomerativeClustering: Fitted Agglomerative Clustering instance
numpy.ndarray: Data used for fitting (PCA-reduced or original)
float: Silhouette score of the clustering (None if n_clusters=1)
Required import:

from sklearn import cluster.
from sklearn import metrics.
Apply_PCA = __import__('1-pca').Apply_PCA."""