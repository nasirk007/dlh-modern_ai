#!/usr/bin/env python3
"""
this module contains functions to scale numeric variables
into numerical value using scikit-learn library.
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Scale numeric variables in a DataFrame using StandardScaler.
    """
    # Scale features to a common range or distribution, important for
    # many ML algorithms sensitive to feature magnitudes.
    # There are two type of scalling one is normal and other is standard.
    # Normalization is used to scale the data between 0 and 1, while
    # standardization is used to scale the data to have a mean of 0
    # and a standard deviation of 1.
    # Transforms features to have mean = 0 and standard deviation = 1,
    # useful for normally distributed features.
    # feature means categorial datatype columns excluding target variable
    # ("Churn") which is also categorical datatype column.
    std_scaler = preprocessing.StandardScaler()
    df[["MonthlyCharges", "TotalCharges"]] = std_scaler.fit_transform(
        df[["MonthlyCharges", "TotalCharges"]])
    return df
