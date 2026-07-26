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
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        dictionary[feature] = p_value
    return dictionary
