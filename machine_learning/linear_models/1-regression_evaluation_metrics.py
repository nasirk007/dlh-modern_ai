#!/usr/bin/env python3
"""module that evaluate linear regression model
"""
import numpy as np
from sklearn import metrics


def evaluation_metrics_for_regression(y_true, y_pred):
    """to compute KPIs of regression model
    """
    mse = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.root_mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return (mse, rmse, mae, r2)
