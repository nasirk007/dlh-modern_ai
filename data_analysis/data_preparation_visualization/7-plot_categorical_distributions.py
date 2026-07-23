#!/usr/bin/env python3
"""
This module visualizes categorical feature distributions.
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Plots bar charts for categorical columns.

    Args:
        df: The pandas DataFrame to analyze.
        columns_to_plot: Optional list of
    categorical columns to plot.

    Returns:
        None
    """
    if columns_to_plot is None:
        columns_to_plot = []
        for column in df.columns:
            if (df[column].dtype == "object" and column != "Churn"):
                columns_to_plot.append(column)
    else:
        columns_to_plot = columns_to_plot

    n_cols, n_rows = 3, (len(columns_to_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    if n_rows == 1:
        axes = [axes]
    for i, column in enumerate(columns_to_plot):
        row = i // n_cols
        col = i % n_cols

        counts = df[column].value_counts()

        axes[row][col].bar(counts.index, counts.values)
        axes[row][col].set_title(column)
        axes[row][col].tick_params(axis="x", rotation=45)

    for i in range(len(columns_to_plot), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()

    return None
