#!/usr/bin/env python3
"""This module visualizes churn class distribution in a DataFrame"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Plots the distribution of the Churn column."""
    plt.figure(figsize=(12, 8))
    counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    plt.bar(counts.index, counts.values, color=["skyblue", "salmon"])
    plt.title("Churn Distribution")
    plt.ylabel("Count")
    plt.show()
    return None
