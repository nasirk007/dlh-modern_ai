#!/usr/bin/env python3
"""First look at the SMS spam data: how many ham and spam messages
there are and how long the messages are."""
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """Draw two plots side by side: ham vs spam counts and a histogram
    of the raw message lengths. Returns nothing."""
    # two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # count ham and spam messages
    counts = df['label'].value_counts()
    x = counts.index
    y = counts.values

    # left plot: bar chart of the counts
    sns.barplot(x=x, y=y, ax=ax1)
    ax1.set_title("Ham vs Spam Counts")
    ax1.set_xlabel("label")
    ax1.set_ylabel("count")

    # length of each raw message
    lengths = df['message'].str.len()

    # right plot: histogram of the lengths
    sns.histplot(lengths, bins=50, ax=ax2)
    ax2.set_title("Histogram of Raw Message Lengths")
    ax2.set_xlabel("length")
    ax2.set_ylabel("count")

    plt.tight_layout()
