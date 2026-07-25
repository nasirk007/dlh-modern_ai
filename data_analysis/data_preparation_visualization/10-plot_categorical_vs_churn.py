#!/usr/bin/env python3
"""
this module visualizes churn rate per category.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    this task isto plot churn rate per category
    use the full DataFrame, and
    group it by the categorical column, e.g in gender 2 categories
    exist, male & female, group all rows of male and female
    separately, the count total female rows (3500), and male rows
    (3532) in the same gender column, in 3500 female group, 700
    include yes in churn column, so 700/3500 = 20% female churn
    likewise, out of total males 1300/3532 = 40% male churned
    in the same fashion, analyse phone service column
    compute churn proportion per group,
    plot that proportion as a bar chart.
    """
    plt.figure(figsize=(12, 8))
    churn_rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    plt.bar(churn_rate.index, churn_rate.values)
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.title(f"Churn Rate by {col}")
    plt.show()
    return None
