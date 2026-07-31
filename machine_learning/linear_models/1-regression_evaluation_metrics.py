#!/usr/bin/env python3
"""module that evaluate regression model
using Scikit-learn.
"""
import numpy as np
from sklearn import metrics


def evaluation_metrics_for_regression(y_true, y_pred):
    """to compute KPIs of regression model,
    including mean absolute error, mean squared error, 
    root mean squared error and R Square.
    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return (mae, mse, rmse, r2)
