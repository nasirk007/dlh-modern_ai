#!/usr/bin/env python3
"""
visualizes the correlation btw continous numeric features.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Plots correlatoin heatmap of numerical features.
    1. Input
    Receive df as the function argument.
    2. Compute correlations
    Use DataFrame method, computes pairwise correlations across
     numeric columns & Result: a square correlation matrix.
    3. Plot heatmap
    Use visualization function that accepts matrix/dataframe,
    & draws color grid. Enable annotations so each cell shows
    its numeric correlation value. Choose coolwarm colormap then.
    Fix the value range with vmin=-1 and vmax=1.
    4. Format appearance
    Add a title like Correlation Matrix.
    Ensure x and y labels are the feature names.
    Keep the plot aspect oriented like the reference image.
    5. Display and return
    Show the plot.
    Return None.
    """
    plt.figure(figsize=(6, 5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="coolwarn", annot=True)
    plt.title("Correlation Matrix")
    plt.show()
    # The correlation matrix can identify potential multicollinearity, which can guide feature selection or combination.
    # tenure and TotalCharges show a strong positive correlation (r = 0.83) but
    # tenure and MonthlyCharges exhibit a weak correlation (r = 0.25).
    return None
