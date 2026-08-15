#!/usr/bin/env python3
"""
This module performs chi-square tests for categorical features, using SciPy.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Perform chi-square tests for categorical features against Churn.
    For each feature other than Churn, compute a contingency table and
    run SciPy's chi2_contingency. The returned dictionary maps each
    feature name to its chi-square p-value.
    A small p-value (commonly < 0.05) suggests the feature and Churn
    are not independent, while a larger p-value suggests the data do not
    provide strong evidence against independence.
    must see 12-practice.ipyn file, how concept applys
    """
    dictionary = {}
    for feature in df.columns:
        if df[feature].dtype == "object" and feature != "Churn":
            table = pd.crosstab(df[feature], df["Churn"])
            # contingency table is a count table that shows how
            # often each combination of two categorical values occurs
            # e.g. gender (male, female) vs churn (Yes, No).
            # run below test and how it work, see 12-practice.ipyn
            # it will compute 4 values and this test need to be done
            # on categorical feature/data/info rather numeric as well
            # each result can be access using index approach [0][1] etc
            chi2, p_value, dof, expected = stats.chi2_contingency(table)
            dictionary[feature] = p_value
    return dictionary
