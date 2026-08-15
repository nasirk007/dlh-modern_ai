#!/usr/bin/env python3
"""
this module visualizes churn rate per continous numerical features.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    this task isto plot churn rate per numerical features
    Compare the distributions of a numeric column for customers
    who churned vs those who did not. The plot should be
    two overlapping histograms (Churn=Yes and Churn=No)
    so you can visually compare where churners concentrate.
    """
    plt.figure(figsize=(12, 8))
    # Data grouping/filtering: split column into two arrays using
    # df["Churn"]=="Yes" & =="No" so each histogram uses only its subgroup.
    # Histogram: counts(ordensity) of values in bins across numeric col.
    # Overlap reveals where one class concentrates relative to the other.
    # Density vs counts: use density=True to compare shapes (probability
    # per bin) even if group sizes differ; otherwise use counts but be
    # aware of different sample sizes.
    # Transparency(alpha): let both histograms be visible.
    Yes_data = df[df["Churn"] == "Yes"][col]
    No_data = df[df["Churn"] == "No"][col]
    plt.hist(
        [Yes_data, No_data],
        bins=30,
        label=["Yes", "No"]
        )
    plt.legend(title="Churn")
    plt.xlabel(col)
    plt.title(f"{col} Distribution by Churn")
    plt.show()
    return None
