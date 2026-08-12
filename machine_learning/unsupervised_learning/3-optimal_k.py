"""3. Choosing the Optimal K for K-Means
Write a function optimal_k(X, max_clusters, random_state) that evaluates K-Means clustering quality using silhouette scores to measure cluster cohesion and separation, and to compute the inertia to be used for the elbow method.

Arguments:

X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
max_clusters (int): Maximum number of clusters to evaluate (>=2)
random_state (int): Random seed for reproducibility
Returns:

list[int]: Evaluated cluster numbers
list[float]: Inertia values corresponding to each cluster number for the elbow method
list[float]: Silhouette scores corresponding to each cluster number for cluster quality evaluation
Required import:

from sklearn import metrics
K_Means = __import__('2-k_means').K_Means."""