#!/usr/bin/env python3
"""
visualizes the correlation btw continous numeric features.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for numeric columns.
    """
    plt.figure(figsize=(6, 5))
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
        )
    plt.title("Correlation Matrix")
    plt.show()
    return None
