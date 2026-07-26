#!/usr/bin/env python3
"""
This module performs chi-square tests for categorical features.
"""
import pandas as pd
from scipy import stats


def ttest_numeric(df):
    """
    Perform t-tests for over numerical data/features.
    three different ttest exist, one sample, two-sample
    and paired ttest.
    large ttest value means, larger diff btw means
    and p-value will confirm whether this diff is
    statistical significant or not.
    """
    dictionary = {}
    for feature in df.columns:
        if df[feature].dtype != "object" and feature != "Churn":
            Yes_Group = df[df["Churn"] == "Yes"][feature].dropna()
            No_Group = df[df["Churn"] == "No"][feature].dropna()
            statistic, p_value = stats.ttest_ind(
                Yes_Group, No_Group, equal_var=False)
            dictionary[feature] = p_value
    return dictionary
