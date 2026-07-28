#!/usr/bin/env python3
"""
this module contains functions to scale numeric variables
into numerical value using scikit-learn, library.
"""
import pandas as pd
from sklearn import preprocessing


def scale_numeric(df):
    """
    Scale numeric variables in a DataFrame using StandardScaler.
    """
    std_scaler = preprocessing.StandardScaler()
    df[["MonthlyCharges", "TotalCharges"]] = std_scaler.fit_transform(
        df[["MonthlyCharges", "TotalCharges"]])
    return df
