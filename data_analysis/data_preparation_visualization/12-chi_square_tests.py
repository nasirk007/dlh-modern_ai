#!/usr/bin/env python3
"""assess independence of categorial feature from churn.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Perform chi-square tests for categorical features against Churn.
    """
    dictionary = {}
    for feature in df.columns:
        if feature == "Churn":
            continue
        table = pd.crosstab(df[feature], df["Churn"])
        test_result = stats.chi2_contingency(table)
        chi2 = test_result[0]
        p_value = test_result[1]
        dof = test_result[2]
        expected = test_result[3]
        dictionary[feature] = p_value
    return dictionary
