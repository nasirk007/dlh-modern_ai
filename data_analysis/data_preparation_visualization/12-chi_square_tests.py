#!/usr/bin/env python3
"""assess independence of categorial feature from churn.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """Perform chi-square tests for categorical features against Churn.
    For each feature other than Churn, compute a contingency table and
    run SciPy's chi2_contingency. The returned dictionary maps each
    feature name to its chi-square p-value.
    A small p-value (commonly < 0.05) suggests the feature and Churn
    are not independent, while a larger p-value suggests the data do not
    provide strong evidence against independence.
    """
    dictionary = {}
    for feature in df.columns:
        if feature == "Churn":
            continue
        # contingency table is a count table that shows how
        # often each combination of two categorical values occurs
        # e.g. gender (male, female) vs churn (Yes, No).
        table = pd.crosstab(df[feature], df["Churn"])
        # run the test to get p-value
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        dictionary[feature] = p_value
    return dictionary
