#!/usr/bin/env python3
"""
This module compares continuous numeric feature distributions by churn.
"""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
        Plots a numeric column distribution
        grouped by Churn.
    """
    plt.figure(figsize=(12, 8))
    churn_no = df[df["Churn"] == "No"][col]
    churn_yes = df[df["Churn"] == "Yes"][col]
    plt.hist(
        [churn_no, churn_yes],
        bins=30,
        label=["No", "Yes"]
        )
    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn")
    plt.show()
    return None
