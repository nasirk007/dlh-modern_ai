#!/usr/bin/env python3
"""this module is to visulize churn class distribution"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """the function will plot customer churn data"""
    plt.figure(figsize=(12, 8))
    count = df["Churn"].value_counts().reindex(["Yes", "No"])
    plt.bar(count.index, count.values, color=["skyblue", "salmon"])
    plt.ylabel("Count")
    plt.title("Churn_Distribution")
    plt.show()
    return None
