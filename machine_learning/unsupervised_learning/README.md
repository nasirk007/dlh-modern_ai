# Unsupervised Learning Module

This module introduces unsupervised learning techniques for discovering structure in unlabeled data. It focuses on feature scaling, dimensionality reduction, clustering, and evaluation, using Scikit-Learn workflows to turn raw data into actionable insights.

## Module Purpose

Unsupervised learning is valuable when labels are unavailable and the goal is to explore data, identify groups, and reduce complexity. This module teaches the principles and practical implementation of PCA, K-Means clustering, and hierarchical clustering for real-world analysis.

## Resources

### Read or watch

- What is unsupervised learning?
- Importance of Feature Scaling
- What Is Principal Components Analysis?
- Principal Component Analysis (PCA): A Step-by-Step Explanation
- In Depth: Principal Component Analysis
- Principal Component Analysis (PCA) with Scikit-Learn
- How to Get Superior Results with Fewer Dimensions?
- In Depth: k-Means Clustering
- Visualizing K-Means Clustering
- Introduction to k-Means Clustering with scikit-learn in Python
- Determine the optimal value of K in K-Means Clustering - ML
- Elbow Method vs. Silhouette Score: which is better?
- Revisiting k-Means: 3 Approaches to Make It Work Better
- Elbow Method for optimal value of k in KMeans
- What is Agglomerative clustering?
- Implementing Agglomerative Clustering using Sklearn
- Agglomerative clustering with and without structure in Scikit Learn
- How to Combine PCA and K-means Clustering in Python?
- Agglomerative Clustering (Dendrograms & PCA)
- Clustering and Principal Component Analysis (PCA) from Sklearn

### References

- `load_wine`
- Preprocessing data
- `StandardScaler`
- Decomposing signals in components
- PCA
- `PCA.fit_transform`
- Clustering
- `KMeans`
- `KMeans.fit`
- `silhouette_score`
- A demo of K-Means clustering on the handwritten digits data
- Selecting the number of clusters with silhouette analysis on KMeans clustering
- `AgglomerativeClustering`

## Learning Objectives

By the end of this project, you should be able to explain the following concepts without using Google:

### General

- What unsupervised learning is
- How unsupervised learning differs from supervised learning
- Why it is important to standardize data before applying clustering algorithms
- What dimensionality reduction is and why it is useful
- What PCA is and how it helps with dimensionality reduction
- What explained variance in PCA is and why it matters
- What K-Means clustering is and how it works
- What cluster centroids are
- What the Elbow Method is and what it is used for
- How to evaluate the quality of clusters
- What the Silhouette Score indicates about clusters
- What hierarchical (Agglomerative) clustering is
- What a dendrogram is and how it helps interpret clusters
- What linkage methods are in hierarchical clustering
- How to visualize clusters in reduced dimensions
- How dimensionality reduction can affect clustering results

## Requirements

- All files will run on Ubuntu 20.04 LTS using `python3` (3.11)
- All files should end with a new line
- The first line of each file must be exactly `#!/usr/bin/env python3`
- A `README.md` file at the root of the project folder is mandatory
- Code must follow `pycodestyle` version `2.14.0`
- All modules must include documentation strings
- All classes must include documentation strings
- All functions must include documentation strings
- All files must be executable
- File lengths may be checked using `wc`
- Required package versions:
  - `numpy==2.0.2`
  - `pandas==2.2.2`
  - `scikit-learn==1.6.1`
  - `matplotlib==3.10.0`
  - `seaborn==0.13.2`
  - `scipy==1.16.0`
  - `pillow==11.3.0`

## Project Workflow

1. Load the dataset and inspect its structure
2. Standardize numeric features with `StandardScaler`
3. Use PCA for dimensionality reduction and review explained variance
4. Apply K-Means clustering to discover groups
5. Evaluate cluster quality with the Elbow Method and silhouette score
6. Apply hierarchical clustering and interpret dendrograms
7. Visualize clustering results in reduced dimensions

## Recommended Task Structure

```text
unsupervised_learning/
├── 0-standardize.py
├── 1-pca.py
├── 2-k_means.py
├── 3-optimal_k.py
├── 4-agglomerative.py
└── README.md
```

## Skills Developed

- Unsupervised learning fundamentals
- Data preprocessing and feature scaling
- Dimensionality reduction using PCA
- Cluster analysis using K-Means and Agglomerative Clustering
- Cluster validation with silhouette score and the Elbow Method
- Interpreting dendrograms and reduced-dimensional plots
- Producing documented, style-compliant Python modules
