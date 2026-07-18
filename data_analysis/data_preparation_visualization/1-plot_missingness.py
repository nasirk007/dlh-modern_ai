#!/usr/bin/env python3
"""visualise missing values in pandas dataframe"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """This task isto plot missing value in datafram."""
    plt.figure(figsize=(12, 8))
    # find missing val and location of missing_value
    missing_row, missing_col = np.where(df.isna())
    # first step isto plot missing value on x and y
    plt.scatter(missing_row, missing_col, marker="|")
    plt.yticks(range(len(df.columns)), df.colummns)
    plt.title("Missingness Plot")
    plt.tight_layout()
    plt.show()
    return None
