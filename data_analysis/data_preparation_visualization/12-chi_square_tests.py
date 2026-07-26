#!/usr/bin/env python3
"""
This module performs chi-square tests for categorical features, using SciPy.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Computes chi-square p-values
for categorical columns against Churn
    """
    dictionary = {}
    for feature in df.columns:
        if df[feature].dtype == "object" and feature != "Churn":
            table = pd.crosstab(df[feature], df["Churn"])

            test_result = stats.chi2_contingency(table)

            chi2 = test_result[0]
            p_value = test_result[1]
            dof = test_result[2]
            expected = test_result[3]

            dictionary[feature] = p_value
    return dictionary
