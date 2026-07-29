#!/usr/bin/env python3
"""module that initiate linear regression model
"""
from sklearn import linear_model


def Linear_Regression():
    """Create an untrained LinearRegression instance
    """
    model = linear_model.LinearRegression()
    # this create linear regression model but we
    # did not trained it yet to make prediction
    return model
