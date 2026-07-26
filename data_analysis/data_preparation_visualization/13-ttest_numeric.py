#!/usr/bin/env python3
"""
This module performs chi-square tests for categorical features, using SciPy.
"""
import pandas as pd
from scipy import stats


def ttest_numeric(df):
    """
    Perform t-tests for over numerical data/features.
    three different ttest exist, one sample, two-sample
    and paired ttest.
    compare one sample mean to know pop mean
    compare mean of two separate groups
    compared
    large ttest value means, larger diff btw means
    and p-value will confirm whether this diff is
    statistical significant
    """
    dictionary = {}
    for feature in df.columns:
        if df[feature].dtype != "object" and feature != "Churn":
            Yes_Group = df[df["Churn"] == "Yes"][feature].dropna()
            No_Group = df[df["Churn"] == "No"][feature].dropna()
            statistic, p_value = stats.ttest_ind(
                Yes_Group, No_Group, equal_var=False)
            # t-test is used when sample size is less than 30 or when
            # population std deviation is unknown, and it uses sample
            # std deviation and adj with uncertainty
            # while z-test rely on ND and used population
            # std deviation in any case
            # eqaul variance assumption of ttest, means variance btw
            # two group are not equal and make it explicit by "False"
            # test will compare mean of Yes_Group with No_Group
            # rather mean of Churn col with mean of tenure col
            dictionary[feature] = p_value
    return dictionary
