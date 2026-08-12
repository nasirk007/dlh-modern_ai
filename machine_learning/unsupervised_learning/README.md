# Unsupervised Learning Module
This module builds a practical introduction to unsupervised machine learning. It is designed to turn unlabeled tabular data into insight by standardizing features, reducing dimensions, grouping similar observations, and validating clusters.

## Why Unsupervised Learning?
Unsupervised learning is used when labels are unavailable. Rather than predicting a known target, it discovers structure, relationships, and hidden groups in data.

### How it differs from supervised learning
- Supervised learning uses labeled examples to learn a mapping to a target variable.
- Unsupervised learning explores unlabeled data to reveal patterns and groupings.
- Supervised models answer "what label should this sample get?" while unsupervised models answer "how is this data organized?"

## Algorithms in This Module

### Feature Standardization
Standardization rescales each feature so it has a mean of 0 and a standard deviation of 1. This step is essential for distance-based and variance-based algorithms.

### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that transforms data into a smaller number of principal components while preserving variance.
- Function: `Apply_PCA(X, n_components, random_state)`
- Output: PCA-transformed data and a fitted PCA instance

### K-Means Clustering
K-Means groups observations into `k` clusters by iteratively assigning points to the nearest centroid and updating each centroid.
- Function: `K_Means(X, n_clusters, random_state)`
- Output: a fitted `KMeans` model

### Optimal K Selection
This task evaluates clustering models with inertia and silhouette scores to identify the most appropriate number of clusters.
- Function: `optimal_k(X, max_clusters, random_state)`
- Output: cluster counts, inertia values, and silhouette scores

### Agglomerative Hierarchical Clustering
Hierarchical clustering builds nested clusters using a linkage criterion. Ward linkage groups data by minimizing variance within clusters.
- Function: `Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True)`
- Output: fitted model, data used for fitting, and silhouette score

## Real-World Business Applications
These techniques are especially valuable in finance and audit workflows:
- Segmenting investment portfolios or client groups using transaction and risk data
- Identifying anomalous accounting entries or fraud signals in audit datasets
- Reducing financial factor sets for valuation models and risk dashboards
- Grouping similar assets for relative valuation, peer benchmarking, or sector analysis
- Detecting unusual investment behavior, outlier issuers, or unexpected market structures

## Project Workflow
1. Load the dataset and inspect its structure
2. Standardize numeric features with `StandardScaler`
3. Apply PCA and inspect explained variance
4. Train K-Means and review cluster centroids
5. Use the Elbow Method and silhouette score to validate cluster count
6. Train Agglomerative Clustering and analyze dendrograms
7. Visualize clustering results in reduced-dimensional space

## Task Overview

### 0. Feature Standardization
`Standardize(X)` scales tabular data so all features contribute proportionally.

### 1. Dimensionality Reduction with PCA
`Apply_PCA(X, n_components, random_state)` performs PCA and returns transformed data and the fitted PCA model.

### 2. Clustering with K-Means
`K_Means(X, n_clusters, random_state)` trains and returns a fitted K-Means model.

### 3. Choosing the Optimal K for K-Means
`optimal_k(X, max_clusters, random_state)` evaluates inertia and silhouette score for cluster counts from 2 to `max_clusters`.

### 4. Agglomerative Hierarchical Clustering
`Agglomerative_Clustering(X, n_clusters, random_state, n_components, use_pca_data=True)` applies optional PCA, fits a Ward linkage model, and returns model, data used, and silhouette score.

## Skills Developed
- Unsupervised learning fundamentals
- Data preprocessing and feature scaling
- Dimensionality reduction with PCA
- Cluster analysis using K-Means and Agglomerative Clustering
- Cluster validation with the Elbow Method and silhouette score
- Interpreting dendrograms and reduced-dimensional visualizations
- Writing documented, style-compliant Python code