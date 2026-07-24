#!/usr/bin/env python3
"""
visualizes the distributions of continuous numerical features
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """Plots distribution for continous numerical columns.
    """
    if columns_to_plot is None:
        columns_to_plot = []
        for col in df.columns:
            # use in instead of == becuase single string
            # cannot be compared with whole list
            if df[col].dtype in ["int64", "float64"]:
                columns_to_plot.append(col)
    # set up the grid and this n_cols variable isto
    # setup now of row of grid rather column and this clarified
    # from syntax of subplot()
    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)
        # fig, axes = plt.subplots(3, 2, ...)
        # Python returns: [[ax1, ax2], [ax3, ax4], [ax5, ax6]] (2D array)
        # Can use: axes[row][col] ==> its okay
        # fig, axes = plt.subplots(1, 2, ...)
        # Python returns: [ax1, ax2]  (1D array - simplified!)
        # Cannot use: axes[row][col] ==> (error!)
        # so we need reshape to access subplot in grid consistently
        # in any condition

    for i, column in enumerate(columns_to_plot):
        # loop through each numerical column using its index and name
        # pick one column under loop, check it first whether it contain missing value
        # remove NaN/None value from that particular columns, this is to make sure
        # subplot does not have gap, meams consistently bar appears to reflect data in them 
        data = df[column].dropna()
        # create histogram on left side, as axes is subplot and i=rowofgrid
        # while j=colofgrid, and here there are 2 col in grid, 0 represent first
        # col which will on left while 1 will represent col on right side of grid
        # create 30 bars from data, data is one numeric column each time loop will run
        # through from start to end, dansity is simply count/frequency/dansity/probability of 
        # continous values in one particular numerical columns, Alpha make bars transparent,
        # while boarder will be colored black
        axes[i, 0].hist(
            data,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor='black'
            )
        # on top of histrogram curve type plot also needed and this curve can be
        # plotted using scientisific & statistical library of python "Scipy"
        # gaussian_kde() calculates a smooth probability curve 
        # KDE = Kernel Density Estimation
        # using numpy array, creates 200 evenly-spaced points from min to max value
        # again i=row in the grid while 0 is column of the grid
        # loop need to move row and col wise for each column curve plot
        # data or values need to draw this dotted line curve is again values from each column
        # and which is under the loop df[column].dropna()
        kde = stats.gaussian_kde(data)
        x_values = np.linspace(data.min(), data.max(), 200)
        axes[i, 0].plot(
            x_values,
            kde(x_values),
            color="red",
            linestyle="--"
            )
        axes[i, 0].set_title(f"{column} Histogram + KDE")

        # this represent the right side plot in the grid
        # box plot normally use percentile to represent the data in each col under loop
        axes[i, 1].boxplot(data, vert=False)
        axes[i, 1].set_title(f"{column} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()

    return None
