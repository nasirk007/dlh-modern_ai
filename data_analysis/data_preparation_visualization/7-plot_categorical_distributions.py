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
    # grid formation with subplot representing each column
    # grid is simply box of many subplots
    # each subplot respresent one column from dataframe
    # This sets up a grid of 3 columns.
    # The number of rows is calculated from 
    # how many columns will be plotted.
    # +2 = n_cols - 1 and it needs to get whole number
    # because 19/3 = 6.1 and not possible that grid may have
    # 6.1 row rather 6 rows only
    n_cols, n_rows = 3, (len(columns_to_plot) + 2) // 3
    # This creates the grid and all the subplots.
    # fig is called grid while subplot are called axes, and they are blank now
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    # to fill each blank subplot with data
    # no of row in the grid if 1, means 1+2/3 = 1
    # so there will be only one subplot, means user just want to
    # see bar chart for one of the column in categorial dataframe
    if n_rows == 1:
        axes = [axes]
    # This loops through each selected categorical column.
    for i, column in enumerate(columns_to_plot):
        # This finds which row current subplot should go into
        # both index/position and value are accessed
        row = i // n_cols
        # This finds which col current subplot should go into.
        col = i % n_cols
        # This counts how many times each category appears in that column
        # this will give both unique categroies and count of unique
        # categories in each columns/axes ,e.g internetservice col has
        # three categories, DSL, No, and NoInternetService and total rows
        # in each column are 7032m breakdown btw these three categories 
        counts = df[column].value_counts()
        # this will draw bar chart for each column
        # axes[row][col] means go to each row of grid and then col
        # pick one particular subplot to prepara its bar chat
        axes[row][col].bar(counts.index, counts.values)
        # This gives the subplot the column name as its title
        axes[row][col].set_title(column)
        # This rotates the category labels on the x-axis by 45 degrees.
        axes[row][col].tick_params(axis="x", rotation=45)

    # This handles any leftover chart in the grid. because row and col
    # created in row 36 for grid needs to cover total categorial columns of 19
    # 19/3 = 6.1 or 19+3-1=21 ==> 21/3 ==> 7 rows and 7th row will have
    # only subplot and rest two needs to be removed  
    # user want to see 7 and there will be 3 rows, and 3rd row should
    # have only one subplot and rest of emprty subplot need to be removed  
    for i in range(len(columns_to_plot), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        #This turns off the axes for empty spaces so they do not appear as blank charts.
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()

    return None
