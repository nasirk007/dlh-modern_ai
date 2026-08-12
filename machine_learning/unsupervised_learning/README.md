# Unsupervised Learning Module

This module teaches unsupervised learning methods for extracting meaningful structure from unlabeled data. It emphasizes feature scaling, dimensionality reduction, clustering, and validation so your work is both rigorous and presentation-ready.

## Module Purpose

Unsupervised learning is essential when labels are unavailable and the goal is to discover groups, reduce dimensionality, or visualize complex datasets. This module focuses on practical Scikit-Learn workflows for PCA, K-Means, and hierarchical clustering.

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
- What is Agglomerative clustering ?
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
- What is unsupervised learning?
- How does unsupervised learning differ from supervised learning?
- Why is it important to standardize data before applying clustering algorithms?
- What is dimensionality reduction and why is it useful?
- What is PCA and how does it help with dimensionality reduction?
- What is explained variance in PCA and why does it matter?
- What is K-Means clustering and how does it work?
- What are cluster centroids?
- What is the Elbow Method and what is it used for?
- How do you evaluate the quality of clusters?
- What does the Silhouette Score indicate about clusters?
- What is hierarchical (Agglomerative) clustering?
- What is a dendrogram and how can it help interpret clusters?
- What are linkage methods in hierarchical clustering?
- How can you visualize clusters in reduced dimensions?
- How can dimensionality reduction affect clustering results?

## Requirements

- All files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version `3.11`)
- All files should end with a new line
- The first line of all files should be exactly `#!/usr/bin/env python3`
- A `README.md` file at the root of the project folder is mandatory
- Your code should use `pycodestyle` style (version `2.14.0`)
- All your modules should have documentation strings
- All your classes should have documentation strings
- All your functions (inside and outside a class) should have documentation strings
- All your files must be executable
- The length of your files may be tested using `wc`

### Required package versions
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
3. Apply PCA and inspect explained variance
4. Train K-Means and review cluster centroids
5. Use the Elbow Method and silhouette score to validate cluster count
6. Train Agglomerative Clustering and analyze dendrograms
7. Visualize clustering results in reduced-dimensional space

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
- Dimensionality reduction with PCA
- Cluster analysis using K-Means and Agglomerative Clustering
- Cluster validation with the Elbow Method and silhouette score
- Interpreting dendrograms and reduced-dimensional visualizations
- Writing documented, style-compliant Python code
